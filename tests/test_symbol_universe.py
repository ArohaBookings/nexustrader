from __future__ import annotations

import unittest

from src.symbol_universe import symbol_family_defaults, symbol_rule_defaults


class SymbolUniverseTests(unittest.TestCase):
    def test_jpy_cross_defaults_use_jpy_tick_geometry(self) -> None:
        defaults = symbol_family_defaults("EURJPY")
        self.assertEqual(defaults["point"], 0.001)
        self.assertEqual(defaults["trade_tick_size"], 0.001)
        self.assertEqual(defaults["trade_tick_value"], 0.9)

    def test_usdjpy_rule_defaults_preserve_three_digit_geometry(self) -> None:
        rule = symbol_rule_defaults("USDJPY")
        self.assertEqual(rule["digits"], 3)
        self.assertEqual(rule["point"], 0.001)
        self.assertEqual(rule["trade_tick_size"], 0.001)
        self.assertEqual(rule["trade_tick_value"], 0.9)
        self.assertEqual(rule["typical_spread_points"], 16)

    def test_non_jpy_major_defaults_remain_standard_forex(self) -> None:
        defaults = symbol_family_defaults("EURUSD")
        self.assertEqual(defaults["point"], 0.0001)
        self.assertEqual(defaults["trade_tick_size"], 0.0001)
        self.assertEqual(defaults["trade_tick_value"], 10.0)
