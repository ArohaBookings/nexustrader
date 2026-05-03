from __future__ import annotations

import unittest

from src.bridge_stop_validation import SymbolRule
from src.trade_management import (
    BrokerSafeModifyInput,
    RetracementManagementInput,
    _profile_overrides,
    build_local_management_plan,
    build_local_trade_plan,
    evaluate_profitable_trade_management,
    smart_manage_trade,
    validate_broker_safe_modify,
)


def _rule(
    *,
    symbol: str = "BTCUSD",
    digits: int = 0,
    tick_size: float = 1.0,
    point: float = 1.0,
    min_stop_points: int = 10,
    freeze_points: int = 0,
) -> SymbolRule:
    return SymbolRule(
        symbol=symbol,
        digits=digits,
        tick_size=tick_size,
        point=point,
        min_stop_points=min_stop_points,
        freeze_points=freeze_points,
        typical_spread_points=10,
        max_slippage_points=50,
        tick_value=1.0,
        contract_size=1.0,
    )


class BrokerSafeModifyTests(unittest.TestCase):
    def test_buy_modify_clamps_to_broker_safe_stop_and_tp(self) -> None:
        result = validate_broker_safe_modify(
            BrokerSafeModifyInput(
                symbol="BTCUSD",
                side="BUY",
                current_bid=100.0,
                current_ask=101.0,
                desired_sl=95.0,
                desired_tp=105.0,
                current_sl=80.0,
                current_tp=130.0,
            ),
            _rule(),
        )

        self.assertTrue(result.valid)
        self.assertTrue(result.actionable)
        self.assertEqual(result.normalized_sl, 90.0)
        self.assertEqual(result.normalized_tp, 110.0)
        self.assertIn("invalid_due_to_stop_level", result.clamp_reasons)

    def test_sell_modify_clamps_to_broker_safe_stop_and_tp(self) -> None:
        result = validate_broker_safe_modify(
            BrokerSafeModifyInput(
                symbol="BTCUSD",
                side="SELL",
                current_bid=100.0,
                current_ask=101.0,
                desired_sl=104.0,
                desired_tp=99.0,
                current_sl=125.0,
                current_tp=80.0,
            ),
            _rule(),
        )

        self.assertTrue(result.valid)
        self.assertTrue(result.actionable)
        self.assertEqual(result.normalized_sl, 111.0)
        self.assertEqual(result.normalized_tp, 91.0)
        self.assertIn("invalid_due_to_stop_level", result.clamp_reasons)

    def test_modify_tracks_freeze_level_clamp_reason(self) -> None:
        result = validate_broker_safe_modify(
            BrokerSafeModifyInput(
                symbol="BTCUSD",
                side="BUY",
                current_bid=100.0,
                current_ask=101.0,
                desired_sl=98.0,
                desired_tp=120.0,
                current_sl=85.0,
                current_tp=130.0,
            ),
            _rule(min_stop_points=0, freeze_points=5),
        )

        self.assertTrue(result.valid)
        self.assertTrue(result.actionable)
        self.assertEqual(result.normalized_sl, 95.0)
        self.assertIn("invalid_due_to_freeze_level", result.clamp_reasons)

    def test_modify_suppresses_no_effective_change(self) -> None:
        result = validate_broker_safe_modify(
            BrokerSafeModifyInput(
                symbol="BTCUSD",
                side="BUY",
                current_bid=100.0,
                current_ask=101.0,
                desired_sl=90.4,
                desired_tp=110.4,
                current_sl=90.0,
                current_tp=110.0,
            ),
            _rule(),
        )

        self.assertTrue(result.valid)
        self.assertFalse(result.actionable)
        self.assertEqual(result.reason, "no_effective_change")

    def test_modify_rejects_missing_live_price(self) -> None:
        result = validate_broker_safe_modify(
            BrokerSafeModifyInput(
                symbol="BTCUSD",
                side="BUY",
                current_bid=0.0,
                current_ask=0.0,
                desired_sl=90.0,
                desired_tp=110.0,
            ),
            _rule(),
        )

        self.assertFalse(result.valid)
        self.assertFalse(result.actionable)
        self.assertEqual(result.reason, "invalid_due_to_missing_live_price")


class SmartManagementFastFailTests(unittest.TestCase):
    def test_btc_be_profit_lock_tightens_as_soon_as_small_profit_prints(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="BTCUSD",
                side="BUY",
                entry_price=66450.0,
                current_price=66454.0,
                sl=66410.0,
                tp=66530.0,
                pnl_r=0.10,
                age_minutes=1.4,
                spread_points=16.0,
                typical_spread_points=14.0,
                volume=0.01,
                min_lot=0.01,
                strategy_family="BTCUSD_PRICE_ACTION_CONTINUATION",
                setup="BTCUSD_PRICE_ACTION_CONTINUATION",
                timeframe="M3",
                session_name="SYDNEY",
                ai_decision="HOLD",
                ai_confidence=0.33,
                runtime_features={
                    "mc_win_rate": 0.84,
                    "m1_ret_1": 4,
                    "m1_ret_3": 7,
                    "m5_ret_1": 5,
                    "m5_ret_3": 8,
                    "m1_momentum_1": 4,
                    "m5_momentum_3": 5,
                    "m5_macd_hist_slope": 0.00011,
                    "m5_slope": 0.00008,
                    "microstructure_composite_score": 0.58,
                },
            )
        )

        self.assertIn(decision.management_action, {"TIGHTEN_STOP", "TRAIL_STOP"})
        self.assertIn(decision.reason, {"btc_be_profit_lock", "btc_ratchet_trail"})
        self.assertGreater(float(decision.tighten_to_price or 0.0), 66450.0)

    def test_btc_micro_profit_recycle_exits_when_small_winner_stalls(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="BTCUSD",
                side="BUY",
                entry_price=66450.0,
                current_price=66454.4,
                sl=66410.0,
                tp=66530.0,
                pnl_r=0.11,
                age_minutes=2.6,
                spread_points=18.0,
                typical_spread_points=14.0,
                volume=0.01,
                min_lot=0.01,
                strategy_family="BTCUSD_PRICE_ACTION_CONTINUATION",
                setup="BTCUSD_PRICE_ACTION_CONTINUATION",
                timeframe="M3",
                session_name="SYDNEY",
                ai_decision="HOLD",
                ai_confidence=0.30,
                runtime_features={
                    "mc_win_rate": 0.79,
                    "m1_ret_1": -2,
                    "m1_ret_3": -4,
                    "m5_ret_1": -3,
                    "m5_ret_3": -5,
                    "m1_momentum_1": -2,
                    "m5_momentum_3": -3,
                    "m5_macd_hist_slope": -0.00008,
                    "m5_slope": -0.00006,
                    "microstructure_composite_score": 0.36,
                    "m5_range_position_20": 0.52,
                },
            )
        )

        self.assertIn(decision.management_action, {"FULL_EXIT", "TRAIL_STOP"})
        self.assertIn(decision.reason, {"btc_micro_profit_recycle", "btc_ratchet_trail"})

    def test_btc_profit_giveback_exit_cuts_after_small_winner_fades(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="BTCUSD",
                side="BUY",
                entry_price=66450.0,
                current_price=66450.8,
                sl=66410.0,
                tp=66530.0,
                pnl_r=0.02,
                age_minutes=2.2,
                spread_points=17.0,
                typical_spread_points=14.0,
                volume=0.01,
                min_lot=0.01,
                strategy_family="BTCUSD_PRICE_ACTION_CONTINUATION",
                setup="BTCUSD_PRICE_ACTION_CONTINUATION",
                timeframe="M3",
                session_name="LONDON",
                mfe_r=0.16,
                ai_decision="HOLD",
                ai_confidence=0.31,
                runtime_features={
                    "m1_ret_1": -2,
                    "m1_ret_3": -5,
                    "m5_ret_1": -3,
                    "m5_ret_3": -6,
                    "m1_momentum_1": -2,
                    "m5_momentum_3": -4,
                    "m5_macd_hist_slope": -0.00008,
                    "m5_slope": -0.00006,
                    "microstructure_composite_score": 0.34,
                    "m5_range_position_20": 0.57,
                },
            )
        )

        self.assertEqual(decision.management_action, "FULL_EXIT")
        self.assertEqual(decision.reason, "btc_profit_giveback_exit")

    def test_xau_grid_fast_fail_exits_weak_negative_probe(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="XAUUSD",
                side="BUY",
                entry_price=4560.24,
                current_price=4559.95,
                sl=4559.10,
                tp=4562.10,
                pnl_r=-0.24,
                age_minutes=0.08,
                spread_points=15.0,
                typical_spread_points=15.0,
                volume=0.01,
                min_lot=0.01,
                setup="XAUUSD_M5_GRID_SCALPER_START",
                timeframe="M3",
                session_name="LONDON",
                ai_decision="HOLD",
                ai_confidence=0.32,
                runtime_features={
                    "m1_ret_1": -8,
                    "m1_ret_3": -11,
                    "m5_ret_1": -6,
                    "m5_ret_3": -10,
                    "m1_momentum_1": -7,
                    "m5_momentum_3": -8,
                    "m5_macd_hist_slope": -0.00012,
                    "m5_slope": -0.00009,
                    "m5_range_position_20": 0.38,
                    "m5_atr_pct_of_avg": 0.94,
                },
            )
        )

        self.assertEqual(decision.management_action, "FULL_EXIT")
        self.assertEqual(decision.reason, "xau_grid_fast_fail")
        self.assertEqual(decision.protection_mode, "xau_grid_fast_fail")

    def test_secondary_breakout_fast_fail_cuts_losing_jpy_lane(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="AUDJPY",
                side="BUY",
                entry_price=97.10,
                current_price=97.02,
                sl=96.90,
                tp=97.50,
                pnl_r=-0.34,
                age_minutes=6.0,
                spread_points=18.0,
                typical_spread_points=12.0,
                volume=0.01,
                min_lot=0.01,
                strategy_family="AUDJPY_ASIA_MOMENTUM_BREAKOUT",
                setup="AUDJPY_ASIA_MOMENTUM_BREAKOUT",
                timeframe="M15",
                ai_decision="HOLD",
                ai_confidence=0.30,
                runtime_features={
                    "m1_ret_1": -4,
                    "m1_ret_3": -9,
                    "m5_ret_1": -7,
                    "m5_ret_3": -13,
                    "m1_momentum_1": -4,
                    "m5_momentum_3": -6,
                    "m5_macd_hist_slope": -0.00008,
                    "m5_slope": -0.00006,
                    "m5_range_position_20": 0.40,
                    "m5_atr_pct_of_avg": 0.90,
                },
            )
        )

        self.assertEqual(decision.management_action, "FULL_EXIT")
        self.assertEqual(decision.reason, "breakout_fast_fail")
        self.assertEqual(decision.protection_mode, "breakout_fast_fail")

    def test_breakout_fast_fail_triggers_earlier_when_no_follow_through(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="NZDJPY",
                side="SELL",
                entry_price=88.20,
                current_price=88.24,
                sl=88.38,
                tp=87.85,
                pnl_r=-0.22,
                age_minutes=1.2,
                spread_points=19.0,
                typical_spread_points=12.0,
                volume=0.01,
                min_lot=0.01,
                strategy_family="NZDJPY_ASIA_MOMENTUM_BREAKOUT",
                setup="NZDJPY_ASIA_MOMENTUM_BREAKOUT",
                timeframe="M15",
                mfe_r=0.10,
                ai_decision="HOLD",
                ai_confidence=0.31,
                runtime_features={
                    "m1_ret_1": 3,
                    "m1_ret_3": 7,
                    "m5_ret_1": 6,
                    "m5_ret_3": 11,
                    "m1_momentum_1": 2,
                    "m5_momentum_3": 4,
                    "m5_macd_hist_slope": 0.00007,
                    "m5_slope": 0.00005,
                    "m5_range_position_20": 0.62,
                    "m5_atr_pct_of_avg": 0.89,
                },
            )
        )

        self.assertEqual(decision.management_action, "FULL_EXIT")
        self.assertEqual(decision.reason, "breakout_fast_fail")


class RetracementManagementTests(unittest.TestCase):
    def test_protected_state_never_loosen_stop_after_lock(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="EURUSD",
                side="BUY",
                entry_price=1.1000,
                current_price=1.1012,
                sl=1.1007,
                tp=1.1030,
                pnl_r=0.55,
                age_minutes=10.0,
                spread_points=8.0,
                typical_spread_points=6.0,
                volume=0.01,
                min_lot=0.01,
                setup="EURUSD_LONDON_EXPANSION_BREAKOUT",
                strategy_family="EURUSD_LONDON_EXPANSION_BREAKOUT",
                timeframe="M15",
                last_trade_state="PROTECTED",
                runtime_features={
                    "mc_win_rate": 0.86,
                    "fractal_persistence_score": 0.70,
                    "compression_expansion_score": 0.62,
                    "dxy_support_score": 0.60,
                    "partial_closes_supported": False,
                },
                learning_brain_bundle={
                    "pair_directives": {
                        "EURUSD": {
                            "management_directives": {
                                "no_loosen_after_protected": True,
                                "early_protect_r": 0.20,
                                "trail_backoff_r": 0.34,
                                "stall_exit_bias": 0.22,
                            }
                        }
                    }
                },
            )
        )

        self.assertEqual(decision.management_action, "HOLD")
        self.assertEqual(decision.reason, "protected_state_no_loosen_hold")
        self.assertEqual(str(decision.details.get("management_state") or ""), "PROTECTED")

    def test_partial_exit_is_rewritten_when_partials_are_not_supported(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="BTCUSD",
                side="BUY",
                entry_price=68000.0,
                current_price=68080.0,
                sl=67920.0,
                tp=68160.0,
                pnl_r=1.0,
                age_minutes=12.0,
                spread_points=12.0,
                typical_spread_points=10.0,
                volume=0.02,
                min_lot=0.01,
                strategy_family="TREND",
                setup="BTC_TREND_SCALP",
                timeframe="M15",
                ai_decision="HOLD",
                ai_confidence=0.52,
                runtime_features={"partial_closes_supported": False},
            )
        )

        self.assertIn(decision.management_action, {"TIGHTEN_STOP", "TRAIL_STOP"})
        self.assertIn(
            decision.protection_mode,
            {"no_partial_profit_lock", "btc_ratchet_trail"},
        )
        self.assertTrue(
            str(decision.reason).endswith("_no_partial_support")
            or str(decision.reason) == "btc_ratchet_trail"
        )

    def test_xau_grid_capture_trail_waits_for_min_age(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="XAUUSD",
                side="BUY",
                entry_price=4560.0,
                current_price=4560.45,
                sl=4558.8,
                tp=4562.4,
                pnl_r=0.375,
                age_minutes=0.08,
                spread_points=14.0,
                typical_spread_points=14.0,
                volume=0.01,
                min_lot=0.01,
                strategy_family="GRID",
                setup="XAUUSD_M5_GRID_SCALPER_START",
                timeframe="M3",
                session_name="LONDON",
                ai_decision="HOLD",
                ai_confidence=0.40,
                runtime_features={
                    "m1_ret_1": 2,
                    "m1_ret_3": 5,
                    "m5_ret_1": 4,
                    "m5_ret_3": 7,
                    "m1_momentum_1": 3,
                    "m5_momentum_3": 5,
                    "m5_macd_hist_slope": 0.00012,
                    "m5_slope": 0.00010,
                    "m5_range_position_20": 0.71,
                    "m5_atr_pct_of_avg": 0.96,
                },
            )
        )

        self.assertNotEqual(decision.reason, "xau_grid_capture_trail")

    def test_xau_grid_weak_launch_fail_exits_stalled_early_loser(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="XAUUSD",
                side="BUY",
                entry_price=4560.0,
                current_price=4559.90,
                sl=4558.8,
                tp=4562.4,
                pnl_r=-0.09,
                age_minutes=0.90,
                spread_points=15.0,
                typical_spread_points=14.0,
                volume=0.01,
                min_lot=0.01,
                strategy_family="GRID",
                setup="XAUUSD_M5_GRID_SCALPER_START",
                timeframe="M3",
                session_name="LONDON",
                mfe_r=0.05,
                ai_decision="HOLD",
                ai_confidence=0.34,
                runtime_features={
                    "m1_ret_1": -2,
                    "m1_ret_3": -5,
                    "m5_ret_1": -3,
                    "m5_ret_3": -6,
                    "m1_momentum_1": -2,
                    "m5_momentum_3": -4,
                    "m5_macd_hist_slope": -0.00006,
                    "m5_slope": -0.00005,
                    "m5_range_position_20": 0.48,
                    "m5_atr_pct_of_avg": 0.93,
                },
            )
        )

        self.assertEqual(decision.management_action, "FULL_EXIT")
        self.assertEqual(decision.reason, "xau_grid_weak_launch_fail")

    def test_xau_grid_micro_profit_recycle_exits_weak_small_winner(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="XAUUSD",
                side="BUY",
                entry_price=4560.0,
                current_price=4560.16,
                sl=4558.8,
                tp=4562.4,
                pnl_r=0.14,
                age_minutes=0.62,
                spread_points=16.0,
                typical_spread_points=14.0,
                volume=0.01,
                min_lot=0.01,
                strategy_family="GRID",
                setup="XAUUSD_M5_GRID_SCALPER_START",
                timeframe="M3",
                session_name="LONDON",
                mfe_r=0.18,
                ai_decision="HOLD",
                ai_confidence=0.34,
                runtime_features={
                    "m1_ret_1": -1,
                    "m1_ret_3": -3,
                    "m5_ret_1": -2,
                    "m5_ret_3": -4,
                    "m1_momentum_1": -2,
                    "m5_momentum_3": -3,
                    "m5_macd_hist_slope": -0.00003,
                    "m5_slope": -0.00002,
                    "m5_range_position_20": 0.61,
                    "m5_atr_pct_of_avg": 1.02,
                },
            )
        )

        self.assertEqual(decision.management_action, "FULL_EXIT")
        self.assertEqual(decision.reason, "xau_grid_micro_profit_recycle")

    def test_xau_grid_profit_giveback_exit_cuts_faded_prime_winner(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="XAUUSD",
                side="BUY",
                entry_price=4560.0,
                current_price=4560.02,
                sl=4558.8,
                tp=4562.4,
                pnl_r=0.01,
                age_minutes=0.75,
                spread_points=15.0,
                typical_spread_points=14.0,
                volume=0.01,
                min_lot=0.01,
                strategy_family="GRID",
                setup="XAUUSD_M5_GRID_SCALPER_START",
                timeframe="M3",
                session_name="LONDON",
                mfe_r=0.20,
                ai_decision="HOLD",
                ai_confidence=0.34,
                runtime_features={
                    "m1_ret_1": -2,
                    "m1_ret_3": -4,
                    "m5_ret_1": -3,
                    "m5_ret_3": -5,
                    "m1_momentum_1": -2,
                    "m5_momentum_3": -4,
                    "m5_macd_hist_slope": -0.00005,
                    "m5_slope": -0.00004,
                    "m5_range_position_20": 0.59,
                    "m5_atr_pct_of_avg": 1.01,
                },
            )
        )

        self.assertEqual(decision.management_action, "FULL_EXIT")
        self.assertEqual(decision.reason, "xau_grid_profit_giveback_exit")

    def test_strategy_family_text_can_drive_range_reversion_template(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="EURUSD",
                side="BUY",
                entry_price=1.1000,
                current_price=1.1016,
                sl=1.0990,
                tp=1.1022,
                pnl_r=1.15,
                age_minutes=18.0,
                spread_points=14.0,
                typical_spread_points=12.0,
                volume=0.02,
                min_lot=0.01,
                strategy_family="EURUSD_RANGE_FADE",
                setup="",
                runtime_features={
                    "m1_ret_1": -35,
                    "m1_ret_3": -60,
                    "m5_ret_1": -30,
                    "m5_ret_3": -55,
                    "m5_range_position_20": 0.48,
                    "m5_atr_pct_of_avg": 0.45,
                    "m5_upper_wick_ratio": 0.58,
                },
            )
        )

        self.assertEqual(decision.management_action, "HOLD")
        self.assertEqual(decision.profit_lock_r, 0.22)

    def test_hold_when_trade_is_not_far_enough_in_profit(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="BTCUSD",
                side="BUY",
                entry_price=68000.0,
                current_price=68020.0,
                sl=67900.0,
                tp=68200.0,
                pnl_r=0.20,
                age_minutes=10.0,
                spread_points=20.0,
                typical_spread_points=15.0,
                volume=0.01,
                min_lot=0.01,
            )
        )

        self.assertEqual(decision.management_action, "HOLD")
        self.assertEqual(decision.reason, "profit_threshold_not_met")
        self.assertIn(decision.trade_state, {"PROVING", "EXIT_READY"})

    def test_tighten_stop_when_profit_exists_but_trend_can_continue(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="BTCUSD",
                side="BUY",
                entry_price=68000.0,
                current_price=68150.0,
                sl=67900.0,
                tp=68200.0,
                pnl_r=0.8,
                age_minutes=60.0,
                spread_points=18.0,
                typical_spread_points=15.0,
                volume=0.01,
                min_lot=0.01,
                ai_decision="MODIFY",
                ai_confidence=0.70,
                runtime_features={
                    "m1_ret_1": 60,
                    "m1_ret_3": 100,
                    "m5_ret_1": 80,
                    "m5_ret_3": 120,
                    "m1_momentum_1": 55,
                    "m5_momentum_3": 75,
                    "m5_macd_hist_slope": 0.0004,
                    "m5_slope": 0.0002,
                    "m5_range_position_20": 0.76,
                    "m5_atr_pct_of_avg": 0.85,
                },
            )
        )

        self.assertEqual(decision.management_action, "TIGHTEN_STOP")
        self.assertEqual(decision.protection_mode, "profit_lock")
        self.assertGreater(float(decision.tighten_to_price or 0.0), 68000.0)
        self.assertGreaterEqual(float(decision.profit_lock_r), 0.10)
        self.assertEqual(decision.trade_state, "PROTECTED")

    def test_fx_trend_like_profit_lock_requires_more_cushion(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="USDJPY",
                side="BUY",
                entry_price=150.00,
                current_price=150.08,
                sl=149.90,
                tp=150.30,
                pnl_r=0.82,
                age_minutes=28.0,
                spread_points=18.0,
                typical_spread_points=12.0,
                volume=0.01,
                min_lot=0.01,
                strategy_family="USDJPY_MOMENTUM_IMPULSE",
                setup="USDJPY_MOMENTUM_IMPULSE",
                runtime_features={
                    "m1_ret_1": 18,
                    "m1_ret_3": 25,
                    "m5_ret_1": 22,
                    "m5_ret_3": 32,
                    "m1_momentum_1": 14,
                    "m5_momentum_3": 20,
                    "m5_macd_hist_slope": 0.0001,
                    "m5_slope": 0.0001,
                    "m5_range_position_20": 0.66,
                    "m5_atr_pct_of_avg": 0.88,
                },
            )
        )

        self.assertIn(decision.management_action, {"HOLD", "TRAIL_STOP"})
        self.assertIn(
            decision.reason,
            {"continuation_still_acceptable", "profit_threshold_not_met", "xau_grid_ratchet_trail"},
        )

    def test_fast_xau_attack_lane_scratches_quickly_when_progress_stalls(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="XAUUSD",
                side="BUY",
                entry_price=2200.0,
                current_price=2200.12,
                sl=2199.4,
                tp=2201.6,
                pnl_r=0.10,
                age_minutes=6.0,
                spread_points=18.0,
                typical_spread_points=14.0,
                volume=0.02,
                min_lot=0.01,
                time_stop_minutes=12,
                timeframe="M3",
                session_name="LONDON",
                strategy_family="GRID",
                setup="XAUUSD_ADAPTIVE_M5_GRID",
                runtime_features={
                    "lane_name": "XAU_LONDON_REENTRY",
                    "execution_timeframe_used": "M3",
                    "fast_execution_profile": "M3_ATTACK",
                    "m5_ret_1": 0.0,
                    "m5_ret_3": 0.0,
                    "m5_range_position_20": 0.54,
                    "m5_atr_pct_of_avg": 0.92,
                },
            )
        )

        self.assertEqual(decision.management_action, "FULL_EXIT")
        self.assertEqual(decision.protection_mode, "time_stop")
        self.assertEqual(decision.reason, "scratch_exit")

    def test_fx_trend_like_profit_lock_does_not_tighten_on_weak_net_after_spread(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="USDJPY",
                side="BUY",
                entry_price=150.00,
                current_price=150.12,
                sl=149.90,
                tp=150.34,
                pnl_r=1.05,
                age_minutes=30.0,
                spread_points=24.0,
                typical_spread_points=12.0,
                volume=0.01,
                min_lot=0.01,
                strategy_family="USDJPY_MOMENTUM_IMPULSE",
                setup="USDJPY_MOMENTUM_IMPULSE",
                runtime_features={
                    "m1_ret_1": 0,
                    "m1_ret_3": 0,
                    "m5_ret_1": 0,
                    "m5_ret_3": 0,
                    "m1_momentum_1": 0,
                    "m5_momentum_3": 0,
                    "m5_macd_hist_slope": 0.0,
                    "m5_slope": 0.0,
                    "m5_range_position_20": 0.56,
                    "m5_atr_pct_of_avg": 0.68,
                },
            )
        )

        self.assertIn(decision.management_action, {"HOLD", "TRAIL_STOP"})
        self.assertIn(
            decision.reason,
            {"continuation_still_acceptable", "profit_threshold_not_met", "xau_grid_ratchet_trail"},
        )

    def test_profit_lock_requires_net_positive_buffer_after_spread(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="BTCUSD",
                side="BUY",
                entry_price=68000.0,
                current_price=68040.0,
                sl=67900.0,
                tp=68200.0,
                pnl_r=0.22,
                age_minutes=35.0,
                spread_points=38.0,
                typical_spread_points=15.0,
                volume=0.01,
                min_lot=0.01,
                ai_decision="MODIFY",
                ai_confidence=0.72,
                runtime_features={
                    "m1_ret_1": 15,
                    "m1_ret_3": 25,
                    "m5_ret_1": 20,
                    "m5_ret_3": 35,
                    "m1_momentum_1": 12,
                    "m5_momentum_3": 18,
                    "m5_macd_hist_slope": 0.0002,
                    "m5_slope": 0.0001,
                    "m5_range_position_20": 0.62,
                    "m5_atr_pct_of_avg": 0.95,
                },
            )
        )

        self.assertEqual(decision.management_action, "HOLD")
        self.assertIn(decision.reason, {"profit_threshold_not_met", "continuation_still_acceptable"})

    def test_trend_partial_capture_triggers_when_spread_drag_is_high_and_continuation_weakens(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="USDJPY",
                side="BUY",
                entry_price=150.00,
                current_price=150.15,
                sl=149.88,
                tp=150.32,
                pnl_r=1.02,
                age_minutes=42.0,
                spread_points=24.0,
                typical_spread_points=12.0,
                volume=0.03,
                min_lot=0.01,
                strategy_family="USDJPY_MOMENTUM_IMPULSE",
                setup="USDJPY_MOMENTUM_IMPULSE",
                runtime_features={
                    "m1_ret_1": 0,
                    "m1_ret_3": 0,
                    "m5_ret_1": 0,
                    "m5_ret_3": 0,
                    "m1_momentum_1": 0,
                    "m5_momentum_3": 0,
                    "m5_macd_hist_slope": 0.0,
                    "m5_slope": 0.0,
                    "m5_range_position_20": 0.60,
                    "m5_atr_pct_of_avg": 0.88,
                    "m5_upper_wick_ratio": 0.40,
                },
            )
        )

        self.assertEqual(decision.management_action, "PARTIAL_EXIT")
        self.assertEqual(decision.protection_mode, "trend_partial_capture")
        self.assertEqual(decision.reason, "trend_partial_capture_on_weakening_continuation")
        self.assertAlmostEqual(decision.close_fraction, 0.5, places=6)

    def test_fx_trend_like_partial_capture_triggers_on_degrading_extension(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="USDJPY",
                side="BUY",
                entry_price=150.00,
                current_price=150.09,
                sl=149.90,
                tp=150.18,
                pnl_r=0.84,
                age_minutes=24.0,
                spread_points=16.0,
                typical_spread_points=12.0,
                volume=0.03,
                min_lot=0.01,
                strategy_family="USDJPY_MOMENTUM_IMPULSE",
                setup="USDJPY_MOMENTUM_IMPULSE",
                runtime_features={
                    "m1_ret_1": 2,
                    "m1_ret_3": 5,
                    "m5_ret_1": 4,
                    "m5_ret_3": 7,
                    "m1_momentum_1": 2,
                    "m5_momentum_3": 4,
                    "m5_macd_hist_slope": -0.00005,
                    "m5_slope": -0.00004,
                    "m5_range_position_20": 0.70,
                    "m5_atr_pct_of_avg": 0.86,
                    "m5_upper_wick_ratio": 0.38,
                },
            )
        )

        self.assertEqual(decision.management_action, "PARTIAL_EXIT")
        self.assertEqual(decision.protection_mode, "trend_partial_capture")
        self.assertEqual(decision.reason, "trend_partial_capture_on_degrading_extension")
        self.assertAlmostEqual(decision.close_fraction, 0.5, places=6)

    def test_partial_exit_when_stall_detected_and_position_size_allows_it(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="BTCUSD",
                side="BUY",
                entry_price=68000.0,
                current_price=68150.0,
                sl=67900.0,
                tp=68200.0,
                pnl_r=1.1,
                age_minutes=100.0,
                spread_points=30.0,
                typical_spread_points=15.0,
                volume=0.03,
                min_lot=0.01,
                ai_decision="CLOSE",
                ai_confidence=0.95,
                timeframe="M5",
                runtime_features={
                    "m1_ret_1": -200,
                    "m1_ret_3": -300,
                    "m5_ret_1": -250,
                    "m5_ret_3": -350,
                    "m1_momentum_1": -160,
                    "m5_momentum_3": -220,
                    "m5_pinbar_bear": 1.0,
                    "m5_engulf_bear": 1.0,
                    "m5_atr_pct_of_avg": 0.2,
                    "m5_upper_wick_ratio": 0.8,
                    "m5_range_position_20": 0.42,
                },
            )
        )

        self.assertEqual(decision.management_action, "PARTIAL_EXIT")
        self.assertEqual(decision.protection_mode, "stall_partial_exit")
        self.assertAlmostEqual(decision.close_fraction, 0.5, places=6)
        self.assertTrue(decision.stall_detected)

    def test_full_close_when_reversal_risk_is_materially_high(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="BTCUSD",
                side="BUY",
                entry_price=68000.0,
                current_price=68190.0,
                sl=67900.0,
                tp=68200.0,
                pnl_r=3.0,
                age_minutes=200.0,
                spread_points=40.0,
                typical_spread_points=15.0,
                volume=0.01,
                min_lot=0.01,
                ai_decision="CLOSE",
                ai_confidence=1.0,
                runtime_features={
                    "m1_ret_1": -500,
                    "m1_ret_3": -500,
                    "m5_ret_1": -500,
                    "m5_ret_3": -500,
                    "m1_momentum_1": -500,
                    "m5_momentum_3": -500,
                    "m5_pinbar_bear": 1.0,
                    "m5_engulf_bear": 1.0,
                    "m5_atr_pct_of_avg": 0.0,
                },
            )
        )

        self.assertEqual(decision.management_action, "FULL_EXIT")
        self.assertEqual(decision.protection_mode, "full_profit_protect")
        self.assertEqual(decision.trade_state, "FORCE_EXIT")

    def test_extend_tp_only_when_continuation_strengthens(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="BTCUSD",
                side="BUY",
                entry_price=68000.0,
                current_price=68190.0,
                sl=67900.0,
                tp=68200.0,
                pnl_r=1.9,
                age_minutes=55.0,
                spread_points=14.0,
                typical_spread_points=15.0,
                volume=0.01,
                min_lot=0.01,
                ai_decision="HOLD",
                ai_confidence=0.82,
                timeframe="M5",
                runtime_features={
                    "m1_ret_1": 120,
                    "m1_ret_3": 180,
                    "m5_ret_1": 150,
                    "m5_ret_3": 240,
                    "m1_momentum_1": 95,
                    "m5_momentum_3": 150,
                    "m5_macd_hist_slope": 0.001,
                    "m5_slope": 0.0005,
                    "m5_range_position_20": 0.94,
                    "m5_atr_pct_of_avg": 0.92,
                },
            )
        )

        self.assertEqual(decision.management_action, "EXTEND_TP")
        self.assertEqual(decision.trade_state, "RUNNER")
        self.assertGreater(float(decision.updated_tp_price or 0.0), 68200.0)
        self.assertGreater(float(decision.tighten_to_price or 0.0), 68000.0)

    def test_time_stop_exits_stale_trade_without_waiting_for_full_stop(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="EURUSD",
                side="BUY",
                entry_price=1.1000,
                current_price=1.1002,
                sl=1.0988,
                tp=1.1025,
                pnl_r=0.05,
                age_minutes=145.0,
                spread_points=16.0,
                typical_spread_points=12.0,
                volume=0.01,
                min_lot=0.01,
                ai_decision="MODIFY",
                ai_confidence=0.55,
                timeframe="M15",
                time_stop_minutes=90,
                runtime_features={
                    "m1_ret_1": -10,
                    "m1_ret_3": -20,
                    "m5_ret_1": -10,
                    "m5_ret_3": -15,
                    "m1_momentum_1": -5,
                    "m5_momentum_3": -8,
                    "m5_upper_wick_ratio": 0.55,
                    "m5_range_position_20": 0.46,
                    "m5_atr_pct_of_avg": 0.25,
                },
            )
        )

        self.assertEqual(decision.management_action, "FULL_EXIT")
        self.assertEqual(decision.reason, "scratch_exit")
        self.assertEqual(decision.protection_mode, "time_stop")

    def test_spread_recovery_partial_triggers_once_spread_is_recouped(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="EURUSD",
                side="BUY",
                entry_price=1.1000,
                current_price=1.10045,
                sl=1.0980,
                tp=1.1020,
                pnl_r=0.45,
                age_minutes=20.0,
                spread_points=2.0,
                typical_spread_points=1.5,
                volume=0.02,
                min_lot=0.01,
                point_size=0.0001,
                timeframe="M5",
                runtime_features={
                    "m1_ret_1": 12,
                    "m1_ret_3": 18,
                    "m5_ret_1": 16,
                    "m5_ret_3": 20,
                    "m5_range_position_20": 0.62,
                    "m5_atr_pct_of_avg": 0.88,
                },
            )
        )

        self.assertEqual(decision.management_action, "PARTIAL_EXIT")
        self.assertEqual(decision.reason, "spread_recovery_partial")
        self.assertAlmostEqual(decision.close_fraction, 0.45, places=6)

    def test_trailing_waits_until_spread_recovery_partial_has_been_bankrolled(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="EURUSD",
                side="BUY",
                entry_price=1.1000,
                current_price=1.1009,
                sl=1.0990,
                tp=1.1020,
                pnl_r=0.90,
                age_minutes=30.0,
                spread_points=60.0,
                typical_spread_points=10.0,
                volume=0.02,
                min_lot=0.01,
                point_size=0.0001,
                spread_recovery_partial_taken=True,
                timeframe="M5",
                ai_decision="HOLD",
                ai_confidence=0.7,
                runtime_features={
                    "m1_ret_1": 20,
                    "m1_ret_3": 26,
                    "m5_ret_1": 24,
                    "m5_ret_3": 32,
                    "m1_momentum_1": 18,
                    "m5_momentum_3": 22,
                    "m5_macd_hist_slope": 0.0002,
                    "m5_slope": 0.00015,
                    "m5_range_position_20": 0.82,
                    "m5_atr_pct_of_avg": 0.86,
                },
            )
        )

        self.assertEqual(decision.management_action, "HOLD")
        self.assertEqual(decision.reason, "continuation_still_acceptable")
        self.assertLess(float(decision.profit_lock_r), 0.60)

    def test_strong_xau_directional_trade_can_upgrade_into_swing_extension(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="XAUUSD",
                side="BUY",
                entry_price=3000.0,
                current_price=3019.0,
                sl=2990.0,
                tp=3020.0,
                pnl_r=1.9,
                age_minutes=85.0,
                spread_points=18.0,
                typical_spread_points=16.0,
                volume=0.02,
                min_lot=0.01,
                point_size=0.01,
                strategy_family="XAUUSD_M15_STRUCTURED_BREAKOUT",
                setup="XAUUSD_M15_STRUCTURED_BREAKOUT",
                timeframe="M15",
                ai_decision="HOLD",
                ai_confidence=0.84,
                runtime_features={
                    "m1_ret_1": 22,
                    "m1_ret_3": 34,
                    "m5_ret_1": 28,
                    "m5_ret_3": 46,
                    "m1_momentum_1": 20,
                    "m5_momentum_3": 30,
                    "m5_macd_hist_slope": 0.0008,
                    "m5_slope": 0.0004,
                    "m5_range_position_20": 0.96,
                    "m5_atr_pct_of_avg": 0.92,
                    "multi_tf_alignment_score": 0.86,
                    "h1_ret_1": 48,
                    "h1_ret_3": 76,
                    "h4_ret_1": 84,
                    "h1_momentum_3": 55,
                    "h1_slope": 0.0005,
                    "h4_slope": 0.0004,
                    "h1_trend_efficiency_16": 0.82,
                    "h4_trend_efficiency_16": 0.78,
                    "structure_cleanliness_score": 0.80,
                },
            )
        )

        self.assertEqual(decision.management_action, "EXTEND_TP")
        self.assertEqual(decision.trade_state, "SWING")
        self.assertEqual(decision.protection_mode, "swing_runner_extension")
        self.assertEqual(decision.reason, "swing_continuation_extend_tp")
        self.assertGreater(float(decision.updated_tp_price or 0.0), 3020.0)

    def test_strong_fx_trend_trade_can_hold_as_swing_when_continuation_is_clean(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="USDJPY",
                side="BUY",
                entry_price=150.00,
                current_price=150.18,
                sl=149.90,
                tp=150.24,
                pnl_r=1.8,
                age_minutes=95.0,
                spread_points=12.0,
                typical_spread_points=12.0,
                volume=0.01,
                min_lot=0.01,
                strategy_family="USDJPY_MOMENTUM_IMPULSE",
                setup="USDJPY_MOMENTUM_IMPULSE",
                timeframe="M15",
                ai_decision="HOLD",
                ai_confidence=0.80,
                runtime_features={
                    "m1_ret_1": 10,
                    "m1_ret_3": 18,
                    "m5_ret_1": 14,
                    "m5_ret_3": 22,
                    "m1_momentum_1": 8,
                    "m5_momentum_3": 14,
                    "m5_macd_hist_slope": 0.00015,
                    "m5_slope": 0.00012,
                    "m5_range_position_20": 0.91,
                    "m5_atr_pct_of_avg": 0.90,
                    "multi_tf_alignment_score": 0.82,
                    "h1_ret_1": 28,
                    "h1_ret_3": 34,
                    "h4_ret_1": 42,
                    "h1_momentum_3": 26,
                    "h1_slope": 0.00014,
                    "h4_slope": 0.00010,
                    "h1_trend_efficiency_16": 0.74,
                    "h4_trend_efficiency_16": 0.72,
                    "structure_cleanliness_score": 0.76,
                },
            )
        )

        self.assertIn(decision.management_action, {"EXTEND_TP", "TRAIL_STOP", "HOLD"})
        self.assertEqual(decision.trade_state, "SWING")
        self.assertIn(
            decision.reason,
            {"swing_continuation_extend_tp", "swing_runner_trailing_update", "swing_continuation_hold"},
        )

    def test_watchdog_forces_profit_protect_when_profitable_hold_goes_stale(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="XAUUSD",
                side="BUY",
                entry_price=3000.0,
                current_price=3002.0,
                sl=2998.0,
                tp=3006.0,
                pnl_r=0.20,
                age_minutes=6.0,
                spread_points=18.0,
                typical_spread_points=16.0,
                volume=0.02,
                min_lot=0.01,
                point_size=0.01,
                strategy_family="GRID",
                setup="XAUUSD_M5_GRID",
                timeframe="M5",
                session_name="LONDON",
                runtime_features={
                    "management_watchdog_force": True,
                    "management_watchdog_reason": "profit_management_cadence_stale",
                    "management_watchdog_stale_seconds": 75,
                    "mc_win_rate": 0.82,
                },
            )
        )

        self.assertEqual(decision.management_action, "TIGHTEN_STOP")
        self.assertEqual(decision.reason, "profit_management_cadence_stale")
        self.assertGreater(float(decision.tighten_to_price or 0.0), 3000.0)

    def test_xau_grid_prime_lane_holds_longer_before_capture_trail_activates(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="XAUUSD",
                side="BUY",
                entry_price=3000.0,
                current_price=3000.9,
                sl=2998.0,
                tp=3004.8,
                pnl_r=0.30,
                age_minutes=3.5,
                spread_points=14.0,
                typical_spread_points=12.0,
                volume=0.02,
                min_lot=0.01,
                point_size=0.01,
                strategy_family="GRID",
                setup="XAUUSD_M5_GRID",
                timeframe="M3",
                session_name="LONDON",
                runtime_features={
                    "mc_win_rate": 0.84,
                    "m1_ret_1": 8,
                    "m1_ret_3": 16,
                    "m5_ret_1": 12,
                    "m5_ret_3": 18,
                    "m1_momentum_1": 7,
                    "m5_momentum_3": 12,
                    "m5_macd_hist_slope": 0.00015,
                    "m5_slope": 0.00012,
                    "m5_range_position_20": 0.74,
                    "m5_atr_pct_of_avg": 0.86,
                    "multi_tf_alignment_score": 0.72,
                    "fractal_persistence_score": 0.70,
                    "compression_expansion_score": 0.66,
                    "execution_minute_quality_score": 0.76,
                },
            )
        )

        self.assertIn(decision.management_action, {"HOLD", "TRAIL_STOP"})
        self.assertIn(
            decision.reason,
            {"continuation_still_acceptable", "profit_threshold_not_met", "xau_grid_ratchet_trail"},
        )

    def test_xau_grid_prime_lane_requires_stronger_profit_before_capture_trail(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="XAUUSD",
                side="BUY",
                entry_price=3000.0,
                current_price=3002.04,
                sl=2998.0,
                tp=3005.4,
                pnl_r=0.68,
                age_minutes=0.58,
                spread_points=14.0,
                typical_spread_points=12.0,
                volume=0.02,
                min_lot=0.01,
                point_size=0.01,
                strategy_family="GRID",
                setup="XAUUSD_M5_GRID",
                timeframe="M3",
                session_name="LONDON",
                mfe_r=0.80,
                runtime_features={
                    "mc_win_rate": 0.87,
                    "m1_ret_1": 10,
                    "m1_ret_3": 16,
                    "m5_ret_1": 12,
                    "m5_ret_3": 18,
                    "m1_momentum_1": 9,
                    "m5_momentum_3": 12,
                    "m5_macd_hist_slope": 0.00018,
                    "m5_slope": 0.00014,
                    "m5_range_position_20": 0.76,
                    "m5_atr_pct_of_avg": 0.88,
                    "multi_tf_alignment_score": 0.76,
                    "fractal_persistence_score": 0.72,
                    "compression_expansion_score": 0.68,
                    "execution_minute_quality_score": 0.80,
                },
            )
        )

        self.assertNotEqual(decision.reason, "xau_grid_capture_trail")

    def test_asia_attack_pair_gets_fast_recycle_profile(self) -> None:
        overrides = _profile_overrides(
            RetracementManagementInput(
                symbol="AUDNZD",
                side="BUY",
                entry_price=1.0800,
                current_price=1.0810,
                sl=1.0785,
                tp=1.0830,
                pnl_r=0.18,
                age_minutes=4.0,
                spread_points=18.0,
                typical_spread_points=16.0,
                volume=0.02,
                min_lot=0.01,
                point_size=0.0001,
                strategy_family="RANGE/REVERSION",
                setup="AUDNZD_COMPRESSION_RELEASE",
                timeframe="M15",
                session_name="TOKYO",
            ),
            symbol_key="AUDNZD",
            family="RANGE/REVERSION",
        )

        self.assertLessEqual(float(overrides["time_stop_factor"]), 0.42)
        self.assertLessEqual(float(overrides["runner_start_r"]), 0.66)
        self.assertLessEqual(float(overrides["scratch_loss_r"]), 0.11)

    def test_super_aggro_fx_trade_promotes_to_swing_with_mc_and_dxy_support(self) -> None:
        decision = evaluate_profitable_trade_management(
            RetracementManagementInput(
                symbol="USDJPY",
                side="BUY",
                entry_price=150.00,
                current_price=150.22,
                sl=149.90,
                tp=150.26,
                pnl_r=2.2,
                age_minutes=110.0,
                spread_points=10.0,
                typical_spread_points=10.0,
                volume=0.01,
                min_lot=0.01,
                strategy_family="USDJPY_MOMENTUM_IMPULSE",
                setup="USDJPY_MOMENTUM_IMPULSE",
                timeframe="M15",
                ai_decision="HOLD",
                ai_confidence=0.86,
                runtime_features={
                    "m1_ret_1": 12,
                    "m1_ret_3": 22,
                    "m5_ret_1": 16,
                    "m5_ret_3": 28,
                    "m1_momentum_1": 10,
                    "m5_momentum_3": 18,
                    "m5_macd_hist_slope": 0.00018,
                    "m5_slope": 0.00014,
                    "m5_range_position_20": 0.94,
                    "m5_atr_pct_of_avg": 0.86,
                    "multi_tf_alignment_score": 0.86,
                    "fractal_persistence_score": 0.80,
                    "compression_expansion_score": 0.72,
                    "mc_win_rate": 0.89,
                    "transition_momentum": 0.36,
                    "dxy_ret_1": 0.0016,
                    "dxy_ret_5": 0.0020,
                    "h1_ret_1": 32,
                    "h1_ret_3": 40,
                    "h4_ret_1": 54,
                    "h1_momentum_3": 30,
                    "h1_slope": 0.00016,
                    "h4_slope": 0.00012,
                    "h1_trend_efficiency_16": 0.78,
                    "h4_trend_efficiency_16": 0.74,
                    "structure_cleanliness_score": 0.80,
                },
            )
        )

        self.assertEqual(decision.trade_state, "SWING")
        self.assertGreaterEqual(float(decision.details.get("mc_win_rate") or 0.0), 0.85)
        self.assertGreaterEqual(float(decision.details.get("dxy_support_score") or 0.0), 0.50)

    def test_build_local_trade_plan_promotes_high_quality_tokyo_jpy_scalp(self) -> None:
        plan = build_local_trade_plan(
            {
                "symbol": "USDJPY",
                "session": "TOKYO",
                "setup": "USDJPY_MOMENTUM_IMPULSE",
                "side": "BUY",
                "probability": 0.70,
                "expected_value_r": 0.48,
                "spread_points": 10.0,
                "point_size": 0.01,
                "min_stop_points": 12.0,
                "mc_win_rate": 0.89,
                "multi_tf_alignment_score": 0.78,
                "fractal_persistence_score": 0.74,
                "compression_expansion_score": 0.66,
                "dxy_support_score": 0.68,
                "learning_brain_bundle": {
                    "quota_catchup_pressure": 0.20,
                    "promoted_patterns": ["USDJPY_MOMENTUM_IMPULSE"],
                    "weak_pair_focus": [],
                    "monte_carlo_pass_floor": 0.82,
                },
            }
        )

        self.assertEqual(str(plan.get("decision") or ""), "TAKE")
        self.assertEqual(str(plan.get("risk_tier") or ""), "HIGH")
        self.assertGreaterEqual(float(plan.get("rr_target") or 0.0), 1.35)

    def test_build_local_trade_plan_uses_pair_directive_for_aggressive_daytrade_bias(self) -> None:
        plan = build_local_trade_plan(
            {
                "symbol": "USOIL",
                "session": "NEW_YORK",
                "setup": "USOIL_LONDON_TREND_EXPANSION",
                "side": "BUY",
                "probability": 0.68,
                "expected_value_r": 0.52,
                "confluence_score": 4.3,
                "spread_points": 8.0,
                "point_size": 0.01,
                "min_stop_points": 20.0,
                "mc_win_rate": 0.90,
                "multi_tf_alignment_score": 0.76,
                "fractal_persistence_score": 0.72,
                "compression_expansion_score": 0.70,
                "dxy_support_score": 0.64,
                "learning_brain_bundle": {
                    "quota_catchup_pressure": 0.65,
                    "promoted_patterns": ["USOIL_LONDON_TREND_EXPANSION"],
                    "pair_directives": {
                        "USOIL": {
                            "trade_horizon_bias": "daytrade",
                            "aggression_multiplier": 1.20,
                            "min_confluence_override": 3.4,
                            "reentry_priority": 0.70,
                        }
                    },
                },
            }
        )

        self.assertEqual(str(plan.get("decision") or ""), "TAKE")
        self.assertGreaterEqual(float(plan.get("rr_target") or 0.0), 1.45)

    def test_build_local_trade_plan_uses_clean_slippage_and_shadow_directive_for_higher_rr(self) -> None:
        plan = build_local_trade_plan(
            {
                "symbol": "BTCUSD",
                "session": "LONDON",
                "setup": "BTCUSD_PRICE_ACTION_CONTINUATION",
                "side": "BUY",
                "probability": 0.72,
                "expected_value_r": 0.55,
                "confluence_score": 4.4,
                "spread_points": 12.0,
                "point_size": 1.0,
                "min_stop_points": 25.0,
                "mc_win_rate": 0.90,
                "multi_tf_alignment_score": 0.82,
                "fractal_persistence_score": 0.78,
                "compression_expansion_score": 0.70,
                "dxy_support_score": 0.66,
                "learning_brain_bundle": {
                    "pair_directives": {
                        "BTCUSD": {
                            "trade_horizon_bias": "scalp",
                            "aggression_multiplier": 1.20,
                            "reentry_priority": 0.80,
                            "slippage_regime": "clean",
                            "shadow_experiment_active": True,
                            "opportunity_capture_gap_r": 0.65,
                            "management_quality_score": 0.48,
                        }
                    }
                },
            }
        )

        self.assertEqual(str(plan.get("decision") or ""), "TAKE")
        self.assertGreaterEqual(float(plan.get("rr_target") or 0.0), 1.70)

    def test_lane_mfe_mae_profile_changes_exit_aggression(self) -> None:
        base_context = {
            "symbol": "BTCUSD",
            "session": "LONDON",
            "setup": "BTCUSD_PRICE_ACTION_CONTINUATION",
            "side": "BUY",
            "probability": 0.72,
            "expected_value_r": 0.50,
            "confluence_score": 4.2,
            "spread_points": 12.0,
            "point_size": 1.0,
            "min_stop_points": 25.0,
            "mc_win_rate": 0.89,
            "multi_tf_alignment_score": 0.80,
            "fractal_persistence_score": 0.76,
            "compression_expansion_score": 0.68,
            "dxy_support_score": 0.64,
        }

        runner_plan = build_local_trade_plan(
            {
                **base_context,
                "learning_brain_bundle": {
                    "pair_directives": {
                        "BTCUSD": {
                            "slippage_regime": "clean",
                            "lane_expectancy_multiplier": 1.16,
                            "hour_expectancy_score": 0.72,
                            "lane_mfe_median_r": 2.10,
                            "lane_mae_median_r": 0.38,
                            "lane_capture_efficiency": 0.66,
                        }
                    }
                },
            }
        )
        stall_plan = build_local_trade_plan(
            {
                **base_context,
                "learning_brain_bundle": {
                    "pair_directives": {
                        "BTCUSD": {
                            "slippage_regime": "rough",
                            "lane_expectancy_multiplier": 0.92,
                            "hour_expectancy_score": 0.36,
                            "lane_mfe_median_r": 0.90,
                            "lane_mae_median_r": 0.72,
                            "lane_capture_efficiency": 0.42,
                        }
                    }
                },
            }
        )

        self.assertGreater(float(runner_plan.get("rr_target") or 0.0), float(stall_plan.get("rr_target") or 0.0))
        self.assertGreater(
            int(runner_plan.get("management_plan", {}).get("time_stop_minutes") or 0),
            int(stall_plan.get("management_plan", {}).get("time_stop_minutes") or 0),
        )

        runner_manage = build_local_management_plan(
            {
                "symbol": "BTCUSD",
                "side": "BUY",
                "setup": "BTCUSD_PRICE_ACTION_CONTINUATION",
                "session": "LONDON",
                "pnl_r": 0.70,
                "age_minutes": 42.0,
                "spread_points": 12.0,
                "typical_spread_points": 10.0,
                "runtime_features": {
                    "mc_win_rate": 0.89,
                    "fractal_persistence_score": 0.76,
                    "multi_tf_alignment_score": 0.80,
                    "compression_expansion_score": 0.68,
                    "dxy_support_score": 0.64,
                },
                "learning_brain_bundle": {
                    "pair_directives": {
                        "BTCUSD": {
                            "slippage_regime": "clean",
                            "lane_expectancy_multiplier": 1.16,
                            "hour_expectancy_score": 0.72,
                            "lane_mfe_median_r": 2.10,
                            "lane_capture_efficiency": 0.66,
                        }
                    }
                },
            }
        )
        stall_manage = build_local_management_plan(
            {
                "symbol": "BTCUSD",
                "side": "BUY",
                "setup": "BTCUSD_PRICE_ACTION_CONTINUATION",
                "session": "LONDON",
                "pnl_r": 0.30,
                "age_minutes": 42.0,
                "spread_points": 12.0,
                "typical_spread_points": 10.0,
                "runtime_features": {
                    "mc_win_rate": 0.89,
                    "fractal_persistence_score": 0.76,
                    "multi_tf_alignment_score": 0.80,
                    "compression_expansion_score": 0.68,
                    "dxy_support_score": 0.64,
                },
                "learning_brain_bundle": {
                    "pair_directives": {
                        "BTCUSD": {
                            "slippage_regime": "rough",
                            "lane_expectancy_multiplier": 0.92,
                            "hour_expectancy_score": 0.36,
                            "lane_mfe_median_r": 0.90,
                            "lane_capture_efficiency": 0.42,
                        }
                    }
                },
            }
        )

        self.assertGreater(
            float(runner_manage.get("management_plan", {}).get("time_stop_minutes") or 0),
            float(stall_manage.get("management_plan", {}).get("time_stop_minutes") or 0),
        )
        self.assertLess(
            float(stall_manage.get("management_plan", {}).get("move_sl_to_be_at_r") or 0.0),
            float(runner_manage.get("management_plan", {}).get("move_sl_to_be_at_r") or 0.0),
        )

    def test_smart_manage_trade_applies_runner_scale_out_after_first_partial(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="USDJPY",
                side="BUY",
                entry_price=150.00,
                current_price=150.15,
                sl=149.90,
                tp=150.24,
                pnl_r=1.10,
                age_minutes=34.0,
                spread_points=10.0,
                typical_spread_points=10.0,
                volume=0.03,
                min_lot=0.01,
                strategy_family="USDJPY_MOMENTUM_IMPULSE",
                setup="USDJPY_MOMENTUM_IMPULSE",
                timeframe="M5",
                spread_recovery_partial_taken=True,
                runner_partial_taken=False,
                learning_brain_bundle={"quota_catchup_pressure": 0.30},
                runtime_features={
                    "m1_ret_1": 6,
                    "m1_ret_3": 9,
                    "m5_ret_1": 8,
                    "m5_ret_3": 12,
                    "m1_momentum_1": 5,
                    "m5_momentum_3": 7,
                    "m5_macd_hist_slope": 0.00008,
                    "m5_slope": 0.00006,
                    "m5_range_position_20": 0.72,
                    "m5_atr_pct_of_avg": 0.90,
                    "multi_tf_alignment_score": 0.70,
                    "fractal_persistence_score": 0.68,
                    "compression_expansion_score": 0.62,
                    "mc_win_rate": 0.86,
                    "transition_momentum": 0.18,
                },
            )
        )

        self.assertEqual(decision.management_action, "PARTIAL_EXIT")
        self.assertEqual(decision.reason, "runner_scale_out_partial")
        self.assertAlmostEqual(float(decision.close_fraction), 0.30, places=6)

    def test_smart_manage_trade_tightens_earlier_when_learning_brain_risk_reduction_is_active(self) -> None:
        decision = smart_manage_trade(
            RetracementManagementInput(
                symbol="EURUSD",
                side="BUY",
                entry_price=1.1000,
                current_price=1.10026,
                sl=1.0990,
                tp=1.1022,
                pnl_r=0.26,
                age_minutes=26.0,
                spread_points=10.0,
                typical_spread_points=10.0,
                volume=0.02,
                min_lot=0.01,
                strategy_family="EURUSD_BREAKOUT",
                setup="EURUSD_BREAKOUT",
                timeframe="M5",
                learning_brain_bundle={
                    "quota_catchup_pressure": 0.25,
                    "risk_reduction_active": True,
                },
                runtime_features={
                    "m1_ret_1": 4,
                    "m1_ret_3": 7,
                    "m5_ret_1": 5,
                    "m5_ret_3": 9,
                    "m1_momentum_1": 3,
                    "m5_momentum_3": 5,
                    "m5_macd_hist_slope": 0.00002,
                    "m5_slope": 0.00003,
                    "m5_range_position_20": 0.62,
                    "m5_atr_pct_of_avg": 0.86,
                    "multi_tf_alignment_score": 0.66,
                    "fractal_persistence_score": 0.64,
                    "compression_expansion_score": 0.57,
                    "mc_win_rate": 0.84,
                    "transition_momentum": 0.15,
                },
            )
        )

        self.assertEqual(decision.management_action, "TIGHTEN_STOP")
        self.assertEqual(decision.reason, "learning_brain_risk_reduction")


if __name__ == "__main__":
    unittest.main()
