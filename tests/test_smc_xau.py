from __future__ import annotations

import unittest

from src.strategies.smc_xau import evaluate_xau_smc_setup


class XauSmcTests(unittest.TestCase):
    def test_smc_confirms_liquidity_sweep_setup(self) -> None:
        decision = evaluate_xau_smc_setup(
            setup="XAUUSD_SMC_SWEEP_RECLAIM",
            reason="liquidity sweep reclaim with rejection",
            side="BUY",
            regime="RANGING",
            probability=0.74,
            expected_value_r=0.9,
            confluence_score=0.62,
            spread_points=28.0,
            spread_cap_points=60.0,
            news_status="clear",
            session_name="LONDON",
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "smc_confirmed")
        self.assertGreater(decision.smc_score, 0.62)

    def test_smc_blocks_when_required_but_no_confirmation(self) -> None:
        decision = evaluate_xau_smc_setup(
            setup="XAUUSD_SMC_ORDERBLOCK",
            reason="weak continuation",
            side="SELL",
            regime="VOLATILE",
            probability=0.55,
            expected_value_r=0.12,
            confluence_score=0.48,
            spread_points=52.0,
            spread_cap_points=60.0,
            news_status="clear",
            session_name="TOKYO",
        )
        self.assertFalse(decision.allowed)
        self.assertIn(decision.reason, {"smc_not_confirmed", "smc_ev_too_low"})


if __name__ == "__main__":
    unittest.main()
