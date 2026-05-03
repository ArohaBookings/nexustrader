from __future__ import annotations

import unittest

from src.bridge_stop_validation import SymbolRule
from src.trade_plan import parse_trade_plan, validate_trade_plan


class TradePlanValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = SymbolRule(
            symbol="XAUUSD",
            digits=2,
            tick_size=0.01,
            point=0.01,
            min_stop_points=10,
            freeze_points=5,
            typical_spread_points=25,
            max_slippage_points=50,
            tick_value=1.0,
            contract_size=100.0,
        )

    def test_daytrade_requires_minimum_rr(self) -> None:
        plan = {
            "decision": "TAKE",
            "setup_type": "daytrade",
            "side": "BUY",
            "sl_points": 120,
            "tp_points": 200,
            "rr_target": 2.0,
            "confidence": 0.9,
            "expected_value_r": 0.6,
            "risk_tier": "HIGH",
            "management_plan": {},
            "notes": "rr too low",
        }
        result = validate_trade_plan(
            plan=plan,
            symbol="XAUUSD",
            side="BUY",
            entry_price=2200.0,
            spread_points=20.0,
            spread_cap_points=40.0,
            symbol_rule=self.rule,
            safety_buffer_points=5,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "daytrade_rr_below_minimum")

    def test_scalp_plan_is_normalized_with_valid_prices(self) -> None:
        plan = {
            "decision": "TAKE",
            "setup_type": "scalp",
            "side": "BUY",
            "sl_points": 150,
            "tp_points": 220,
            "rr_target": 1.5,
            "confidence": 0.7,
            "expected_value_r": 0.4,
            "risk_tier": "NORMAL",
            "management_plan": {"trail_method": "atr"},
            "notes": "ok",
        }
        result = validate_trade_plan(
            plan=plan,
            symbol="XAUUSD",
            side="BUY",
            entry_price=2200.0,
            spread_points=20.0,
            spread_cap_points=40.0,
            symbol_rule=self.rule,
            safety_buffer_points=5,
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "validated")
        normalized = result.normalized_plan
        self.assertGreater(float(normalized["tp_price"]), 2200.0)
        self.assertLess(float(normalized["sl_price"]), 2200.0)
        self.assertGreaterEqual(float(normalized["rr_target"]), 1.0)

    def test_parse_trade_plan_rejects_unknown_decision(self) -> None:
        parsed = parse_trade_plan({"decision": "GO", "setup_type": "scalp", "side": "BUY"})
        self.assertIsNone(parsed)


if __name__ == "__main__":
    unittest.main()
