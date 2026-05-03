from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import pandas as pd
    from src.ai_gate import AIGate, AIGateDecision
    from src.strategy_engine import SignalCandidate
    _HAS_AI_DEPS = True
    _SKIP_REASON = ""
except ModuleNotFoundError as exc:
    pd = None  # type: ignore
    AIGate = None  # type: ignore
    AIGateDecision = None  # type: ignore
    SignalCandidate = None  # type: ignore
    _HAS_AI_DEPS = False
    _SKIP_REASON = f"missing dependency: {exc.name}"


@unittest.skipUnless(_HAS_AI_DEPS, _SKIP_REASON)
class AIGateTests(unittest.TestCase):
    def _build_gate(self) -> AIGate:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            return AIGate(
                scorer_path=root / "missing_scorer.pkl",
                value_path=root / "missing_value.pkl",
                risk_modulator_path=root / "missing_risk.pkl",
                schema_path=root / "missing_schema.json",
                min_probabilities={
                    "DEFAULT": 0.58,
                    "TREND_CONTINUATION": 0.60,
                    "BREAKOUT_RETEST": 0.58,
                    "RANGE_REVERSAL": 0.55,
                    "EV": 0.10,
                },
                enabled=True,
                live_remote_enabled=False,
            )

    def test_fallback_approves_strong_trend_candidate(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "h1_trend_score": 0.9,
                "m15_trend_score": 0.7,
                "m5_macd_hist_slope": 0.03,
                "m5_atr_pct_of_avg": 1.0,
            }
        )
        candidate = SignalCandidate("sig-1", "TREND_CONTINUATION", "BUY", 0.64, "test", 1.5, 1.8)
        decision = gate.evaluate(candidate, row, "TRENDING", consecutive_losses=0)

        self.assertTrue(decision.approved)
        self.assertGreaterEqual(decision.probability, 0.60)
        self.assertGreaterEqual(decision.tp_r, 1.5)

    def test_approve_order_requires_risk_and_news(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "h1_trend_score": 0.9,
                "m15_trend_score": 0.7,
                "m5_macd_hist_slope": 0.03,
                "m5_atr_pct_of_avg": 1.0,
                "m5_spread_ratio_20": 1.0,
            }
        )
        candidate = SignalCandidate("sig-2", "TREND_CONTINUATION", "BUY", 0.68, "test", 1.5, 1.8)

        approved, payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="TRENDING",
            risk_summary={"approved": True, "reason": "approved", "consecutive_losses": 0, "spread_points": 20, "max_spread_points": 35, "max_slippage_points": 10, "portfolio_size_multiplier": 1.0, "requires_ai_override": False},
            news_summary={"safe": False, "reason": "blocked_high_usd", "source": "cache"},
            open_positions=[],
            account_state={"equity": 1000.0},
        )

        self.assertFalse(approved)
        self.assertIn("news_not_safe", payload["rationale"])

    def test_approve_order_emits_final_execution_params(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "h1_trend_score": 0.95,
                "m15_trend_score": 0.75,
                "m5_macd_hist_slope": 0.04,
                "m5_atr_pct_of_avg": 1.0,
                "m5_spread_ratio_20": 1.0,
            }
        )
        candidate = SignalCandidate("sig-3", "TREND_CONTINUATION", "BUY", 0.70, "test", 1.5, 1.8)

        approved, payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="TRENDING",
            risk_summary={"approved": True, "reason": "approved", "consecutive_losses": 0, "spread_points": 20, "max_spread_points": 35, "max_slippage_points": 10, "portfolio_size_multiplier": 1.0, "requires_ai_override": False},
            news_summary={"safe": True, "reason": "clear", "source": "provider"},
            open_positions=[],
            account_state={"equity": 1000.0},
        )

        self.assertTrue(approved)
        self.assertTrue(payload["approve"])
        self.assertIn("recommended_sl_atr_mult", payload)
        self.assertIn("recommended_tp_r", payload)
        self.assertIn("trailing_enabled", payload)
        self.assertIn("partial_close_enabled", payload)

    def test_news_caution_allows_only_stronger_setup(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "h1_trend_score": 0.95,
                "m15_trend_score": 0.75,
                "m5_macd_hist_slope": 0.04,
                "m5_atr_pct_of_avg": 1.0,
                "m5_spread_ratio_20": 1.0,
            }
        )
        weak = SignalCandidate("sig-caution-weak", "TREND_CONTINUATION", "BUY", 0.60, "test", 1.5, 1.8, confluence_score=2.0)
        strong = SignalCandidate("sig-caution-strong", "TREND_CONTINUATION", "BUY", 0.72, "test", 1.5, 1.8, confluence_score=4.0)

        approved_weak, payload_weak = gate.approve_order(
            candidate=weak,
            features=row,
            regime="TRENDING",
            risk_summary={"approved": True, "reason": "approved", "consecutive_losses": 0, "spread_points": 20, "max_spread_points": 35, "max_slippage_points": 10, "portfolio_size_multiplier": 1.0, "requires_ai_override": False},
            news_summary={"safe": True, "state": "NEWS_CAUTION", "reason": "news_caution_provider_unavailable", "source": "cache", "caution_probability_floor": 0.68, "caution_confluence_floor": 0.70},
            open_positions=[],
            account_state={"equity": 1000.0},
        )
        approved_strong, payload_strong = gate.approve_order(
            candidate=strong,
            features=row,
            regime="TRENDING",
            risk_summary={"approved": True, "reason": "approved", "consecutive_losses": 0, "spread_points": 20, "max_spread_points": 35, "max_slippage_points": 10, "portfolio_size_multiplier": 1.0, "requires_ai_override": False},
            news_summary={"safe": True, "state": "NEWS_CAUTION", "reason": "news_caution_provider_unavailable", "source": "cache", "caution_probability_floor": 0.68, "caution_confluence_floor": 0.70},
            open_positions=[],
            account_state={"equity": 1000.0},
        )

        self.assertFalse(approved_weak)
        self.assertIn("news_caution_gate", payload_weak["rationale"])
        self.assertTrue(approved_strong)
        self.assertTrue(payload_strong["approve"])

    def test_approve_order_uses_precomputed_gate_without_re_evaluating(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "h1_trend_score": 0.95,
                "m15_trend_score": 0.75,
                "m5_macd_hist_slope": 0.04,
                "m5_atr_pct_of_avg": 1.0,
                "m5_spread_ratio_20": 1.0,
            }
        )
        candidate = SignalCandidate("sig-3b", "TREND_CONTINUATION", "BUY", 0.70, "test", 1.5, 1.8)
        gate.evaluate = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("evaluate should not be called"))  # type: ignore[method-assign]

        approved, payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="TRENDING",
            risk_summary={"approved": True, "reason": "approved", "consecutive_losses": 0, "spread_points": 20, "max_spread_points": 35, "max_slippage_points": 10, "portfolio_size_multiplier": 1.0, "requires_ai_override": False},
            news_summary={"safe": True, "reason": "clear", "source": "provider"},
            open_positions=[],
            account_state={"equity": 1000.0},
            precomputed_gate=AIGateDecision(True, 0.72, 0.28, 1.0, 1.0, 1.8, "precomputed"),
        )

        self.assertTrue(approved)
        self.assertTrue(payload["approve"])

    def test_adaptive_feedback_can_tighten_probability_gate(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "h1_trend_score": 0.75,
                "m15_trend_score": 0.65,
                "m5_macd_hist_slope": 0.02,
                "m5_atr_pct_of_avg": 1.0,
                "m5_spread_ratio_20": 1.0,
            }
        )
        candidate = SignalCandidate("sig-4", "TREND_CONTINUATION", "BUY", 0.62, "test", 1.5, 1.8)

        approved, payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="TRENDING",
            risk_summary={
                "approved": True,
                "reason": "approved",
                "consecutive_losses": 0,
                "spread_points": 20,
                "max_spread_points": 35,
                "max_slippage_points": 10,
                "portfolio_size_multiplier": 1.0,
                "requires_ai_override": False,
                "adaptive_samples": 20,
                "adaptive_win_rate": 0.30,
                "adaptive_avg_r": -0.25,
                "adaptive_recent_loss_streak": 3,
            },
            news_summary={"safe": True, "reason": "clear", "source": "provider"},
            open_positions=[],
            account_state={"equity": 1000.0},
        )

        self.assertFalse(approved)
        self.assertIn("final_gate_conservative_reject", payload["rationale"])

    def test_authenticity_risk_degrades_probability_for_crowded_social_news(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "h1_trend_score": 0.8,
                "m15_trend_score": 0.7,
                "m5_macd_hist_slope": 0.02,
                "m5_atr_pct_of_avg": 1.0,
                "m5_spread_ratio_20": 1.0,
            }
        )
        candidate = SignalCandidate("sig-auth", "TREND_CONTINUATION", "BUY", 0.61, "test", 1.5, 1.8)

        _, clean_payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="TRENDING",
            risk_summary={
                "approved": True,
                "reason": "approved",
                "consecutive_losses": 0,
                "spread_points": 20,
                "max_spread_points": 35,
                "max_slippage_points": 10,
                "portfolio_size_multiplier": 1.0,
                "requires_ai_override": False,
            },
            news_summary={
                "safe": True,
                "reason": "clear",
                "source": "provider",
                "source_confidence": 0.95,
                "authenticity_risk": 0.0,
                "sentiment_extreme": 0.0,
                "crowding_bias": "neutral",
            },
            open_positions=[],
            account_state={"equity": 1000.0},
        )

        _, crowded_payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="TRENDING",
            risk_summary={
                "approved": True,
                "reason": "approved",
                "consecutive_losses": 0,
                "spread_points": 20,
                "max_spread_points": 35,
                "max_slippage_points": 10,
                "portfolio_size_multiplier": 1.0,
                "requires_ai_override": False,
            },
            news_summary={
                "safe": True,
                "reason": "clear",
                "source": "social",
                "source_confidence": 0.2,
                "authenticity_risk": 0.9,
                "sentiment_extreme": 0.9,
                "crowding_bias": "bullish",
            },
            open_positions=[],
            account_state={"equity": 1000.0},
        )

        self.assertLess(float(crowded_payload["probability"]), float(clean_payload["probability"]))

    def test_session_probability_offset_can_reject_borderline_trade(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "h1_trend_score": 0.0,
                "m15_trend_score": 0.0,
                "m5_macd_hist_slope": 0.0,
                "m5_atr_pct_of_avg": 1.0,
                "m5_spread_ratio_20": 1.0,
            }
        )
        candidate = SignalCandidate("sig-session", "TREND_CONTINUATION", "BUY", 0.60, "test", 1.5, 1.8)

        approved, payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="TRENDING",
            risk_summary={
                "approved": True,
                "reason": "approved",
                "consecutive_losses": 0,
                "spread_points": 18,
                "max_spread_points": 35,
                "max_slippage_points": 10,
                "portfolio_size_multiplier": 1.0,
                "requires_ai_override": False,
                "session_probability_offset": 0.03,
                "rule_confluence_score": 3.0,
                "rule_confluence_required": 5.0,
            },
            news_summary={"safe": True, "reason": "clear", "source": "provider"},
            open_positions=[],
            account_state={"equity": 1000.0},
        )

        self.assertFalse(approved)
        self.assertIn("final_gate_conservative_reject", payload["rationale"])

    def test_news_alignment_and_dxy_support_raise_quality_and_misalignment_cuts_it(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "h1_trend_score": 0.85,
                "m15_trend_score": 0.70,
                "m5_macd_hist_slope": 0.03,
                "m5_atr_pct_of_avg": 1.0,
                "m5_spread_ratio_20": 1.0,
                "dxy_support_score": 0.72,
            }
        )
        candidate = SignalCandidate("sig-align", "TREND_CONTINUATION", "BUY", 0.63, "test", 1.5, 1.8)

        approved_aligned, aligned_payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="TRENDING",
            risk_summary={
                "approved": True,
                "reason": "approved",
                "consecutive_losses": 0,
                "spread_points": 18,
                "max_spread_points": 35,
                "max_slippage_points": 10,
                "portfolio_size_multiplier": 1.0,
                "requires_ai_override": False,
                "dxy_support_score": 0.72,
            },
            news_summary={
                "safe": True,
                "reason": "clear",
                "source": "provider",
                "bias_direction": "bullish",
                "bias_confidence": 0.85,
                "source_confidence": 0.95,
                "authenticity_risk": 0.0,
                "sentiment_extreme": 0.1,
                "crowding_bias": "neutral",
            },
            open_positions=[],
            account_state={"equity": 1000.0},
        )

        approved_misaligned, misaligned_payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="TRENDING",
            risk_summary={
                "approved": True,
                "reason": "approved",
                "consecutive_losses": 0,
                "spread_points": 18,
                "max_spread_points": 35,
                "max_slippage_points": 10,
                "portfolio_size_multiplier": 1.0,
                "requires_ai_override": False,
                "dxy_support_score": 0.30,
            },
            news_summary={
                "safe": True,
                "reason": "clear",
                "source": "provider",
                "bias_direction": "bearish",
                "bias_confidence": 0.85,
                "source_confidence": 0.95,
                "authenticity_risk": 0.0,
                "sentiment_extreme": 0.1,
                "crowding_bias": "neutral",
            },
            open_positions=[],
            account_state={"equity": 1000.0},
        )

        self.assertTrue(approved_aligned)
        self.assertGreater(float(aligned_payload["probability"]), float(misaligned_payload["probability"]))
        self.assertFalse(approved_misaligned)

    def test_slippage_quality_and_hour_expectancy_can_promote_or_reject_same_trade(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "h1_trend_score": 0.78,
                "m15_trend_score": 0.68,
                "m5_macd_hist_slope": 0.02,
                "m5_atr_pct_of_avg": 1.0,
                "m5_spread_ratio_20": 1.0,
                "dxy_support_score": 0.64,
            }
        )
        candidate = SignalCandidate("sig-fill", "TREND_CONTINUATION", "BUY", 0.61, "test", 1.5, 1.8)
        precomputed = AIGateDecision(True, 0.61, 0.22, 1.0, 1.0, 1.8, "precomputed")

        approved_clean, clean_payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="TRENDING",
            risk_summary={
                "approved": True,
                "reason": "approved",
                "consecutive_losses": 0,
                "spread_points": 18,
                "max_spread_points": 35,
                "max_slippage_points": 10,
                "portfolio_size_multiplier": 1.0,
                "requires_ai_override": False,
                "dxy_support_score": 0.64,
                "execution_quality_score": 0.84,
                "spread_quality_score": 0.80,
                "slippage_quality_score": 0.82,
                "execution_quality_state": "GOOD",
                "execution_spread_ema_points": 12.0,
                "hour_expectancy_score": 0.72,
                "lane_expectancy_multiplier": 1.14,
            },
            news_summary={"safe": True, "reason": "clear", "source": "provider", "bias_direction": "bullish", "bias_confidence": 0.75, "source_confidence": 0.9, "authenticity_risk": 0.0},
            open_positions=[],
            account_state={"equity": 1000.0},
            precomputed_gate=precomputed,
        )

        approved_rough, rough_payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="TRENDING",
            risk_summary={
                "approved": True,
                "reason": "approved",
                "consecutive_losses": 0,
                "spread_points": 18,
                "max_spread_points": 35,
                "max_slippage_points": 10,
                "portfolio_size_multiplier": 1.0,
                "requires_ai_override": False,
                "dxy_support_score": 0.64,
                "execution_quality_score": 0.32,
                "spread_quality_score": 0.38,
                "slippage_quality_score": 0.34,
                "execution_quality_state": "DEGRADED",
                "execution_spread_ema_points": 26.0,
                "hour_expectancy_score": 0.35,
                "lane_expectancy_multiplier": 0.90,
            },
            news_summary={"safe": True, "reason": "clear", "source": "provider", "bias_direction": "bullish", "bias_confidence": 0.75, "source_confidence": 0.9, "authenticity_risk": 0.0},
            open_positions=[],
            account_state={"equity": 1000.0},
            precomputed_gate=precomputed,
        )

        self.assertTrue(approved_clean)
        self.assertFalse(approved_rough)
        self.assertGreater(float(clean_payload["probability"]), float(rough_payload["probability"]))

    def test_adaptive_min_floor_can_allow_starved_range_trade(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "h1_trend_score": 0.0,
                "m15_trend_score": 0.0,
                "m5_macd_hist_slope": 0.0,
                "m5_atr_pct_of_avg": 1.0,
                "m5_spread_ratio_20": 1.0,
            }
        )
        candidate = SignalCandidate("sig-starved", "RANGE_REVERSAL", "BUY", 0.56, "test", 1.4, 1.8)

        approved, payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="RANGING",
            risk_summary={
                "approved": True,
                "reason": "approved",
                "consecutive_losses": 0,
                "spread_points": 10,
                "max_spread_points": 35,
                "max_slippage_points": 10,
                "portfolio_size_multiplier": 1.0,
                "requires_ai_override": False,
                "session_probability_offset": -0.05,
                "adaptive_min_probability_floor": 0.50,
                "rule_confluence_score": 4.0,
                "rule_confluence_required": 4.0,
            },
            news_summary={"safe": True, "reason": "clear", "source": "provider"},
            open_positions=[],
            account_state={"equity": 1000.0},
        )

        self.assertTrue(approved)
        self.assertTrue(payload.get("approve"))

    def test_fallback_pass_allows_narrow_ai_miss_when_confluence_strong(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "h1_trend_score": 0.0,
                "m15_trend_score": 0.0,
                "m5_macd_hist_slope": 0.0,
                "m5_atr_pct_of_avg": 1.0,
                "m5_spread_ratio_20": 1.0,
            }
        )
        candidate = SignalCandidate("sig-fallback", "TREND_CONTINUATION", "BUY", 0.61, "test", 1.5, 1.8)

        approved, payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="TRENDING",
            risk_summary={
                "approved": True,
                "reason": "approved",
                "consecutive_losses": 0,
                "spread_points": 18,
                "max_spread_points": 35,
                "max_slippage_points": 10,
                "portfolio_size_multiplier": 1.0,
                "requires_ai_override": False,
                "session_probability_offset": 0.02,
                "rule_confluence_score": 5.0,
                "rule_confluence_required": 5.0,
            },
            news_summary={"safe": True, "reason": "clear", "source": "provider"},
            open_positions=[],
            account_state={"equity": 1000.0},
        )

        self.assertTrue(approved)
        self.assertTrue(payload.get("fallback_pass"))
        self.assertLess(float(payload.get("size_multiplier", 1.0)), 1.0)

    def test_local_trade_plan_builder_returns_local_brain_mode(self) -> None:
        gate = self._build_gate()
        plan, mode = gate.propose_trade_plan(
            {
                "symbol": "USDJPY",
                "timeframe": "M5",
                "setup": "USDJPY_MOMENTUM_IMPULSE",
                "side": "BUY",
                "probability": 0.69,
                "expected_value_r": 0.42,
                "spread_points": 12.0,
                "point_size": 0.01,
                "min_stop_points": 12.0,
                "session": "TOKYO",
                "mc_win_rate": 0.88,
                "multi_tf_alignment_score": 0.76,
                "fractal_persistence_score": 0.72,
                "compression_expansion_score": 0.62,
                "dxy_support_score": 0.66,
                "learning_brain_bundle": {
                    "quota_catchup_pressure": 0.25,
                    "promoted_patterns": ["USDJPY_MOMENTUM_IMPULSE"],
                    "weak_pair_focus": [],
                    "monte_carlo_pass_floor": 0.82,
                },
            }
        )

        self.assertEqual(mode, "local_brain")
        self.assertEqual(str(plan.get("decision") or ""), "TAKE")
        self.assertEqual(str(plan.get("risk_tier") or ""), "HIGH")

    def test_remote_schema_parser_rejects_invalid_payload(self) -> None:
        parsed, error = AIGate._parse_remote_trade_payload({"approve": True, "confidence": 0.5, "direction": "UP"})
        self.assertIsNone(parsed)
        self.assertIsNotNone(error)

    def test_local_approve_order_reports_local_brain_mode(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "h1_trend_score": 0.8,
                "m15_trend_score": 0.6,
                "m5_macd_hist_slope": 0.02,
                "m5_atr_pct_of_avg": 1.0,
                "m5_spread_ratio_20": 1.0,
            }
        )
        candidate = SignalCandidate("sig-fallback-local", "TREND_CONTINUATION", "BUY", 0.68, "test", 1.5, 1.8)
        approved, payload = gate.approve_order(
            candidate=candidate,
            features=row,
            regime="TRENDING",
            risk_summary={
                "approved": True,
                "reason": "approved",
                "consecutive_losses": 0,
                "spread_points": 12,
                "max_spread_points": 35,
                "max_slippage_points": 10,
                "portfolio_size_multiplier": 1.0,
                "requires_ai_override": False,
                "rule_confluence_score": 5.0,
                "rule_confluence_required": 5.0,
            },
            news_summary={"safe": True, "reason": "clear", "source": "provider"},
            open_positions=[],
            account_state={"equity": 1000.0},
        )
        self.assertTrue(approved)
        self.assertIn("rationale", payload)
        self.assertIsInstance(approved, bool)
        self.assertEqual(payload.get("ai_mode"), "local_brain")

    def test_live_remote_disabled_never_calls_remote_trade_plan(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            gate = AIGate(
                scorer_path=root / "missing_scorer.pkl",
                value_path=root / "missing_value.pkl",
                risk_modulator_path=root / "missing_risk.pkl",
                schema_path=root / "missing_schema.json",
                min_probabilities={
                    "DEFAULT": 0.58,
                    "TREND_CONTINUATION": 0.60,
                    "BREAKOUT_RETEST": 0.58,
                    "RANGE_REVERSAL": 0.55,
                    "EV": 0.10,
                },
                enabled=True,
                live_remote_enabled=False,
            )
            gate._openai.trade_plan = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("remote trade_plan should not be called"))  # type: ignore[attr-defined]

            plan, mode = gate.propose_trade_plan(
                {
                    "symbol": "EURUSD",
                    "timeframe": "M5",
                    "setup": "EURUSD_LONDON_BREAKOUT",
                    "side": "BUY",
                    "probability": 0.71,
                    "expected_value_r": 0.42,
                    "spread_points": 12.0,
                    "point_size": 0.0001,
                    "min_stop_points": 80.0,
                    "regime": "TRENDING",
                    "session": "LONDON",
                    "news_status": "clear",
                    "omega_regime": "BALANCED",
                    "entry_price": 1.0850,
                    "min_probability": 0.56,
                }
            )

            self.assertEqual(mode, "local_brain")
            self.assertEqual(str(plan.get("decision") or ""), "TAKE")

    def test_undertrade_fix_mode_relaxes_for_tokyo_native_pair_using_meta_symbol(self) -> None:
        gate = self._build_gate()
        row = pd.Series(
            {
                "symbol": "AUDJPY",
                "session_name": "TOKYO",
                "h1_trend_score": 0.8,
                "m15_trend_score": 0.6,
                "m5_macd_hist_slope": 0.03,
                "m5_atr_pct_of_avg": 1.0,
            }
        )
        candidate = SignalCandidate(
            "sig-audjpy-undertrade",
            "TREND_CONTINUATION",
            "BUY",
            0.62,
            "test",
            1.5,
            1.8,
            confluence_score=3.6,
            meta={
                "symbol_key": "AUDJPY",
                "session_name": "TOKYO",
                "session_native_pair": True,
                "undertrade_fix_mode": True,
                "soft_burst_target_10m": 5,
                "quality_tier": "A",
                "throughput_recovery_active": True,
                "frequency_catchup_pressure": 0.6,
            },
        )

        decision = gate.evaluate(candidate, row, "TRENDING", consecutive_losses=0)

        self.assertTrue(decision.approved)
        self.assertIn("source=", decision.reason)


if __name__ == "__main__":
    unittest.main()
