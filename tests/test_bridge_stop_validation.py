from __future__ import annotations

import unittest

from src.bridge_stop_validation import (
    StopValidationInput,
    estimate_loss_usd,
    load_symbol_rules,
    resolve_symbol_rule,
    validate_and_normalize_stops,
)


class BridgeStopValidationTests(unittest.TestCase):
    def test_xau_buy_stops_are_pushed_out_and_normalized(self) -> None:
        rules = load_symbol_rules(None)
        rule = resolve_symbol_rule("XAUUSD", rules)
        result = validate_and_normalize_stops(
            StopValidationInput(
                symbol="XAUUSD",
                side="BUY",
                entry_price=2200.0,
                sl=2199.95,
                tp=2200.05,
                spread_points=40,
                safety_buffer_points=5,
            ),
            rule,
        )

        self.assertTrue(result.valid)
        self.assertIsNotNone(result.normalized_sl)
        self.assertIsNotNone(result.normalized_tp)
        assert result.normalized_sl is not None
        assert result.normalized_tp is not None
        self.assertLess(result.normalized_sl, 2200.0)
        self.assertGreater(result.normalized_tp, 2200.0)
        self.assertGreaterEqual(result.sl_distance_points, result.min_required_points - 1e-6)

    def test_xau_sell_stops_are_on_correct_side(self) -> None:
        rules = load_symbol_rules(None)
        rule = resolve_symbol_rule("XAUUSD", rules)
        result = validate_and_normalize_stops(
            StopValidationInput(
                symbol="XAUUSD",
                side="SELL",
                entry_price=2200.0,
                sl=2199.0,
                tp=2201.0,
                spread_points=35,
            ),
            rule,
        )

        self.assertTrue(result.valid)
        assert result.normalized_sl is not None
        assert result.normalized_tp is not None
        self.assertGreater(result.normalized_sl, 2200.0)
        self.assertLess(result.normalized_tp, 2200.0)

    def test_estimated_loss_uses_tick_value(self) -> None:
        rules = load_symbol_rules(None)
        rule = resolve_symbol_rule("XAUUSD", rules)
        loss = estimate_loss_usd(entry_price=2200.0, stop_price=2199.0, lot=0.01, rule=rule)
        self.assertGreater(loss, 0.0)

    def test_symbol_alias_resolves_to_xau_rule(self) -> None:
        rules = load_symbol_rules(None)
        base_rule = resolve_symbol_rule("XAUUSD", rules)
        alias_rule = resolve_symbol_rule("XAUUSDm", rules)
        self.assertEqual(base_rule.symbol, alias_rule.symbol)
        self.assertEqual(base_rule.tick_size, alias_rule.tick_size)

    def test_nas_stop_normalization_respects_tick_size(self) -> None:
        rules = load_symbol_rules(None)
        rule = resolve_symbol_rule("US100", rules)
        result = validate_and_normalize_stops(
            StopValidationInput(
                symbol="US100",
                side="BUY",
                entry_price=18325.4,
                sl=18325.1,
                tp=18325.7,
                spread_points=26,
            ),
            rule,
        )
        self.assertTrue(result.valid)
        assert result.normalized_sl is not None
        assert result.normalized_tp is not None
        self.assertAlmostEqual((result.normalized_sl * 10) % 1, 0.0, places=7)
        self.assertAlmostEqual((result.normalized_tp * 10) % 1, 0.0, places=7)

    def test_oil_stop_normalization_respects_tick_size(self) -> None:
        rules = load_symbol_rules(None)
        rule = resolve_symbol_rule("XTIUSD", rules)
        result = validate_and_normalize_stops(
            StopValidationInput(
                symbol="XTIUSD",
                side="SELL",
                entry_price=75.42,
                sl=75.40,
                tp=75.41,
                spread_points=30,
            ),
            rule,
        )
        self.assertTrue(result.valid)
        assert result.normalized_sl is not None
        assert result.normalized_tp is not None
        self.assertGreater(result.normalized_sl, 75.42)
        self.assertLess(result.normalized_tp, 75.42)

    def test_oil_alias_uso_resolves_to_oil_rule(self) -> None:
        rules = load_symbol_rules(None)
        base_rule = resolve_symbol_rule("USOIL", rules)
        alias_rule = resolve_symbol_rule("USO", rules)
        self.assertEqual(base_rule.symbol, alias_rule.symbol)
        self.assertEqual(base_rule.tick_size, alias_rule.tick_size)

    def test_fx_buy_stops_use_live_bid_ask_geometry(self) -> None:
        rules = load_symbol_rules(None)
        rule = resolve_symbol_rule("EURUSD", rules)
        result = validate_and_normalize_stops(
            StopValidationInput(
                symbol="EURUSD",
                side="BUY",
                entry_price=1.16200,
                sl=1.16170,
                tp=1.16235,
                spread_points=12,
                live_bid=1.16190,
                live_ask=1.16200,
                safety_buffer_points=5,
            ),
            rule,
        )

        self.assertTrue(result.valid)
        self.assertTrue(result.market_geometry_used)
        assert result.normalized_sl is not None
        assert result.normalized_tp is not None
        min_gap = max(float(rule.min_stop_points), float(rule.freeze_points)) + 5.0
        self.assertLessEqual(result.normalized_sl, 1.16190 - (min_gap * rule.point) + 1e-9)
        self.assertGreaterEqual(result.normalized_tp, 1.16200 + (min_gap * rule.point) - 1e-9)

    def test_xau_buy_stops_are_clamped_against_live_market(self) -> None:
        rules = load_symbol_rules(None)
        rule = resolve_symbol_rule("XAUUSD", rules)
        result = validate_and_normalize_stops(
            StopValidationInput(
                symbol="XAUUSD",
                side="BUY",
                entry_price=5052.30,
                sl=5050.80,
                tp=5056.83,
                spread_points=30,
                live_bid=5052.00,
                live_ask=5052.30,
                safety_buffer_points=5,
            ),
            rule,
        )

        self.assertTrue(result.valid)
        self.assertTrue(result.market_geometry_used)
        assert result.normalized_sl is not None
        min_gap = max(float(rule.min_stop_points), float(rule.freeze_points)) + 5.0
        self.assertLessEqual(result.normalized_sl, 5052.00 - (min_gap * rule.point) + 1e-9)

    def test_audjpy_alias_resolves_to_audjpy_rule(self) -> None:
        rules = load_symbol_rules(None)
        base_rule = resolve_symbol_rule("AUDJPY", rules)
        alias_rule = resolve_symbol_rule("AUDJPYm", rules)

        self.assertEqual(base_rule.symbol, alias_rule.symbol)
        self.assertEqual(base_rule.tick_size, alias_rule.tick_size)
        self.assertEqual(base_rule.typical_spread_points, 95)

    def test_audnzd_buy_stops_use_loaded_cross_rule_geometry(self) -> None:
        rules = load_symbol_rules(None)
        rule = resolve_symbol_rule("AUDNZD", rules)
        result = validate_and_normalize_stops(
            StopValidationInput(
                symbol="AUDNZD",
                side="BUY",
                entry_price=1.08140,
                sl=1.08090,
                tp=1.08220,
                spread_points=165,
                live_bid=1.08120,
                live_ask=1.08140,
                safety_buffer_points=5,
            ),
            rule,
        )

        self.assertTrue(result.valid)
        self.assertTrue(result.market_geometry_used)
        self.assertEqual(rule.typical_spread_points, 140)
        assert result.normalized_sl is not None
        assert result.normalized_tp is not None
        self.assertLess(result.normalized_sl, 1.08120)
        self.assertGreater(result.normalized_tp, 1.08140)


if __name__ == "__main__":
    unittest.main()
