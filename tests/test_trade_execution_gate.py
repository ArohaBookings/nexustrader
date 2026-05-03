from __future__ import annotations

import unittest

from src.trade_execution_gate import validate_trade_executable


class TradeExecutionGateTests(unittest.TestCase):
    def test_micro_min_lot_tolerance_allows_small_account_trade(self) -> None:
        decision = validate_trade_executable(
            account_equity=50.0,
            symbol="XAUUSD",
            lot=0.01,
            stop_distance=1.2,
            contract_size=100.0,
            tick_size=0.01,
            tick_value=1.0,
            min_lot=0.01,
            risk_budget_usd=0.5,
            trade_plan_risk_cap_usd=0.5,
            micro_account_equity_threshold=500.0,
            micro_min_risk_usd=0.5,
            micro_risk_pct=0.005,
            micro_min_lot_risk_multiplier=4.0,
        )
        self.assertTrue(decision.executable)
        self.assertEqual(decision.status, "EXECUTABLE")
        self.assertEqual(decision.reason, "micro_min_lot_tolerance")

    def test_executable_blocked_when_lot_below_min(self) -> None:
        decision = validate_trade_executable(
            account_equity=50.0,
            symbol="NAS100",
            lot=0.001,
            stop_distance=10.0,
            contract_size=1.0,
            tick_size=0.1,
            tick_value=1.0,
            min_lot=0.01,
            risk_budget_usd=1.0,
            trade_plan_risk_cap_usd=1.0,
        )
        self.assertFalse(decision.executable)
        self.assertEqual(decision.status, "EXECUTION_BLOCK")
        self.assertEqual(decision.reason, "lot_below_min_or_margin_too_low")

    def test_executable_blocked_when_risk_budget_exceeded_without_tolerance(self) -> None:
        decision = validate_trade_executable(
            account_equity=5000.0,
            symbol="XAUUSD",
            lot=0.5,
            stop_distance=3.0,
            contract_size=100.0,
            tick_size=0.01,
            tick_value=1.0,
            min_lot=0.01,
            risk_budget_usd=10.0,
            trade_plan_risk_cap_usd=10.0,
            micro_account_equity_threshold=500.0,
        )
        self.assertFalse(decision.executable)
        self.assertEqual(decision.reason, "trade_plan_risk_exceeded")

    def test_broker_min_lot_tolerance_allows_small_over_cap_trade(self) -> None:
        decision = validate_trade_executable(
            account_equity=82.86,
            symbol="USOIL",
            lot=1.0,
            stop_distance=2.67,
            contract_size=1.0,
            tick_size=0.01,
            tick_value=0.017057,
            min_lot=1.0,
            risk_budget_usd=4.0,
            trade_plan_risk_cap_usd=4.0,
            micro_account_equity_threshold=500.0,
            micro_min_risk_usd=0.5,
            micro_risk_pct=0.005,
            micro_min_lot_risk_multiplier=4.0,
        )

        self.assertTrue(decision.executable)
        self.assertEqual(decision.reason, "micro_min_lot_tolerance")


if __name__ == "__main__":
    unittest.main()
