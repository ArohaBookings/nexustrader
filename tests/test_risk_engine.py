from __future__ import annotations

from datetime import datetime, timezone
import unittest

from src.risk_engine import RiskEngine, RiskInputs, TradeStats, detect_funded_account_mode


class RiskEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RiskEngine()
        self.now = datetime(2026, 3, 4, 14, 0, tzinfo=timezone.utc)

    def _base(self) -> RiskInputs:
        return RiskInputs(
            symbol="XAUUSD",
            mode="DEMO",
            live_enabled=False,
            live_allowed=False,
            current_time=self.now,
            spread_points=20,
            entry_price=2200.0,
            stop_price=2198.5,
            tp_price=2202.25,
            equity=1000.0,
            account_balance=1000.0,
            margin_free=900.0,
            open_positions=0,
            open_positions_symbol=0,
            same_direction_positions=0,
            session_multiplier=1.0,
            symbol_point=0.01,
            contract_size=100.0,
            volume_min=0.01,
            volume_max=10.0,
            volume_step=0.01,
            requested_risk_pct=0.0025,
            hard_risk_cap=0.005,
            max_positions=4,
            max_positions_per_symbol=2,
            max_daily_loss=0.02,
            circuit_breaker_daily_loss=0.03,
            max_drawdown_kill=0.05,
            absolute_drawdown_hard_stop=0.08,
            max_spread_points=35,
            atr_current=1.5,
            atr_average=1.0,
            atr_spike_multiple=2.0,
            volatility_pause_minutes=30,
            regime="TRENDING",
            ai_probability=0.65,
            ai_size_multiplier=1.0,
            portfolio_size_multiplier=1.0,
            recent_trades_last_hour=0,
            max_trades_per_hour=6,
            use_kelly=False,
            kelly_fraction=0.25,
            use_fixed_lot=False,
            fixed_lot=0.01,
            stats=TradeStats(),
            weekend_trading_allowed=False,
        )

    def test_rejects_live_without_explicit_gate(self) -> None:
        payload = self._base()
        payload.mode = "LIVE"
        decision = self.engine.evaluate(payload)
        self.assertFalse(decision.approved)
        self.assertIn("live_blocked", decision.reason)

    def test_rejects_when_spread_too_wide(self) -> None:
        payload = self._base()
        payload.spread_points = 40
        decision = self.engine.evaluate(payload)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "spread_too_wide")

    def test_btc_weekend_spread_override_allows_liveable_crypto_weekend_spread(self) -> None:
        payload = self._base()
        payload.symbol = "BTCUSD"
        payload.current_time = datetime(2026, 3, 28, 6, 0, tzinfo=timezone.utc)
        payload.weekend_trading_allowed = True
        payload.spread_points = 1707.0
        payload.max_spread_points = 60.0
        payload.spread_atr_reference_points = 40.0
        payload.entry_price = 66483.99
        payload.stop_price = 66450.00
        payload.tp_price = 66540.00
        payload.symbol_point = 0.01
        payload.contract_size = 1.0

        decision = self.engine.evaluate(payload)

        self.assertNotEqual(decision.reason, "spread_too_wide")

    def test_low_equity_weekend_btc_override_bypasses_spread_atr_guard(self) -> None:
        payload = self._base()
        payload.symbol = "BTCUSD"
        payload.current_time = datetime(2026, 3, 28, 6, 45, tzinfo=timezone.utc)
        payload.weekend_trading_allowed = True
        payload.equity = 48.38
        payload.account_balance = 42.65
        payload.margin_free = 48.38
        payload.requested_risk_pct = 0.01
        payload.hard_risk_cap = 0.02
        payload.contract_size = 1.0
        payload.symbol_point = 0.01
        payload.entry_price = 66483.99
        payload.stop_price = 66439.60
        payload.tp_price = 66540.00
        payload.spread_points = 1707.0
        payload.max_spread_points = 60.0
        payload.spread_atr_reference_points = 40.0
        payload.trade_quality_score = 0.82
        payload.execution_quality_score = 0.82
        payload.execution_minute_quality_score = 0.69
        payload.session_quality_score = 0.69
        payload.ai_probability = 0.83
        payload.expected_value_r = 1.01
        payload.confluence_score = 4.05
        payload.candidate_monte_carlo_win_rate = 0.83
        payload.microstructure_composite_score = 0.30
        payload.news_state = "NEWS_CAUTION"
        payload.session_priority_multiplier = 0.9
        payload.session_native_pair = False

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved)
        self.assertNotEqual(decision.reason, "low_equity_spread_atr_guard")

    def test_triggers_and_respects_volatility_pause(self) -> None:
        payload = self._base()
        payload.atr_current = 2.5
        decision = self.engine.evaluate(payload)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "atr_spike_pause")

        second = self.engine.evaluate(self._base())
        self.assertFalse(second.approved)
        self.assertIn("volatility_pause_until", second.reason)

    def test_approves_and_sizes_within_caps(self) -> None:
        decision = self.engine.evaluate(self._base())
        self.assertTrue(decision.approved)
        self.assertGreaterEqual(decision.volume, 0.01)
        self.assertLessEqual(decision.risk_pct, 0.005)

    def test_low_equity_mc_floor_blocks_thin_candidates(self) -> None:
        payload = self._base()
        payload.equity = 120.0
        payload.account_balance = 120.0
        payload.margin_free = 120.0
        payload.requested_risk_pct = 0.02
        payload.hard_risk_cap = 0.02
        payload.max_spread_points = 35.0
        payload.spread_atr_reference_points = 20.0
        payload.candidate_monte_carlo_win_rate = 0.84

        decision = self.engine.evaluate(payload)

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "low_equity_mc_floor")

    def test_low_equity_xau_grid_override_allows_strong_prime_candidate(self) -> None:
        payload = self._base()
        payload.equity = 54.11
        payload.account_balance = 50.0
        payload.margin_free = 54.11
        payload.requested_risk_pct = 0.01
        payload.hard_risk_cap = 0.02
        payload.max_spread_points = 35.0
        payload.spread_points = 14.0
        payload.spread_atr_reference_points = 20.0
        payload.candidate_monte_carlo_win_rate = 0.83
        payload.trade_quality_score = 0.64
        payload.execution_quality_score = 0.74
        payload.execution_minute_quality_score = 0.72
        payload.session_quality_score = 0.66
        payload.microstructure_composite_score = 0.58
        payload.lead_lag_alignment_score = 0.12
        payload.setup = "XAUUSD_M5_GRID_SCALPER_START"
        payload.news_state = "NEWS_CAUTION"

        decision = self.engine.evaluate(payload)

        self.assertNotEqual(decision.reason, "low_equity_mc_floor")

    def test_low_equity_xau_grid_override_is_more_permissive_for_live_bootstrap_prime_setup(self) -> None:
        payload = self._base()
        payload.equity = 54.11
        payload.account_balance = 50.0
        payload.margin_free = 54.11
        payload.requested_risk_pct = 0.01
        payload.hard_risk_cap = 0.02
        payload.max_spread_points = 35.0
        payload.spread_points = 15.0
        payload.spread_atr_reference_points = 20.0
        payload.low_equity_monte_carlo_floor = 0.85
        payload.candidate_monte_carlo_win_rate = 0.79
        payload.trade_quality_score = 0.57
        payload.execution_quality_score = 0.62
        payload.execution_minute_quality_score = 0.58
        payload.session_quality_score = 0.60
        payload.microstructure_composite_score = 0.50
        payload.lead_lag_alignment_score = 0.08
        payload.setup = "XAUUSD_M5_GRID_SCALPER_START"
        payload.news_state = "NEWS_CAUTION"

        decision = self.engine.evaluate(payload)

        self.assertNotEqual(decision.reason, "low_equity_mc_floor")

    def test_low_equity_xau_grid_override_allows_strong_tokyo_candidate(self) -> None:
        payload = self._base()
        payload.current_time = datetime(2026, 3, 27, 5, 0, tzinfo=timezone.utc)
        payload.equity = 54.11
        payload.account_balance = 50.0
        payload.margin_free = 54.11
        payload.requested_risk_pct = 0.01
        payload.hard_risk_cap = 0.02
        payload.max_spread_points = 35.0
        payload.spread_points = 16.0
        payload.spread_atr_reference_points = 20.0
        payload.low_equity_monte_carlo_floor = 0.88
        payload.candidate_monte_carlo_win_rate = 0.83
        payload.trade_quality_score = 0.89
        payload.execution_quality_score = 0.71
        payload.execution_minute_quality_score = 0.70
        payload.session_quality_score = 0.63
        payload.microstructure_composite_score = 0.52
        payload.lead_lag_alignment_score = 0.04
        payload.setup = "XAUUSD_M5_GRID_SCALPER_START"
        payload.news_state = "NEWS_CAUTION"

        decision = self.engine.evaluate(payload)

        self.assertNotEqual(decision.reason, "low_equity_mc_floor")

    def test_low_equity_xau_grid_override_bypasses_mild_spread_guard_for_strong_tokyo_candidate(self) -> None:
        payload = self._base()
        payload.current_time = datetime(2026, 3, 27, 5, 0, tzinfo=timezone.utc)
        payload.equity = 26.89
        payload.account_balance = 22.78
        payload.margin_free = 26.89
        payload.requested_risk_pct = 0.01
        payload.hard_risk_cap = 0.02
        payload.max_spread_points = 35.0
        payload.spread_points = 24.2
        payload.spread_atr_reference_points = 20.0
        payload.low_equity_monte_carlo_floor = 0.88
        payload.candidate_monte_carlo_win_rate = 0.76
        payload.trade_quality_score = 0.80
        payload.execution_quality_score = 0.74
        payload.execution_minute_quality_score = 0.70
        payload.session_quality_score = 0.64
        payload.microstructure_composite_score = 0.50
        payload.lead_lag_alignment_score = 0.02
        payload.setup = "XAUUSD_M5_GRID_SCALPER_START"
        payload.news_state = "NEWS_SAFE"

        decision = self.engine.evaluate(payload)

        self.assertNotEqual(decision.reason, "low_equity_spread_atr_guard")

    def test_low_equity_xau_grid_override_bypasses_prime_spread_guard_for_strong_attack_candidate(self) -> None:
        payload = self._base()
        payload.current_time = datetime(2026, 3, 27, 8, 20, tzinfo=timezone.utc)
        payload.equity = 26.76
        payload.account_balance = 22.65
        payload.margin_free = 26.76
        payload.requested_risk_pct = 0.01
        payload.hard_risk_cap = 0.02
        payload.max_spread_points = 35.0
        payload.spread_points = 16.0
        payload.spread_atr_reference_points = 9.0
        payload.low_equity_monte_carlo_floor = 0.88
        payload.candidate_monte_carlo_win_rate = 0.78
        payload.trade_quality_score = 0.86
        payload.execution_quality_score = 0.92
        payload.execution_minute_quality_score = 0.83
        payload.session_quality_score = 0.81
        payload.microstructure_composite_score = 0.52
        payload.lead_lag_alignment_score = 0.18
        payload.setup = "XAUUSD_M5_GRID_SCALPER_START"
        payload.news_state = "NEWS_CAUTION"
        payload.session_native_pair = True
        payload.session_priority_multiplier = 1.2

        decision = self.engine.evaluate(payload)

        self.assertNotEqual(decision.reason, "low_equity_spread_atr_guard")

    def test_low_equity_xau_grid_override_accepts_high_quality_prime_candidate_with_wider_spread(self) -> None:
        payload = self._base()
        payload.current_time = datetime(2026, 3, 27, 8, 45, tzinfo=timezone.utc)
        payload.equity = 26.76
        payload.account_balance = 22.65
        payload.margin_free = 26.76
        payload.requested_risk_pct = 0.01
        payload.hard_risk_cap = 0.02
        payload.max_spread_points = 35.0
        payload.spread_points = 19.8
        payload.spread_atr_reference_points = 9.0
        payload.low_equity_monte_carlo_floor = 0.88
        payload.candidate_monte_carlo_win_rate = 0.78
        payload.trade_quality_score = 0.83
        payload.execution_quality_score = 1.0
        payload.execution_minute_quality_score = 0.70
        payload.session_quality_score = 0.95
        payload.ai_probability = 0.82
        payload.expected_value_r = 1.64
        payload.confluence_score = 5.0
        payload.microstructure_composite_score = 0.30
        payload.lead_lag_alignment_score = 0.04
        payload.setup = "XAUUSD_M5_GRID_SCALPER_START"
        payload.news_state = "NEWS_VOLATILE"

        decision = self.engine.evaluate(payload)

        self.assertNotEqual(decision.reason, "low_equity_spread_atr_guard")

    def test_low_equity_xau_grid_override_keeps_blocking_weak_tokyo_candidate(self) -> None:
        payload = self._base()
        payload.current_time = datetime(2026, 3, 27, 5, 0, tzinfo=timezone.utc)
        payload.equity = 54.11
        payload.account_balance = 50.0
        payload.margin_free = 54.11
        payload.requested_risk_pct = 0.01
        payload.hard_risk_cap = 0.02
        payload.max_spread_points = 35.0
        payload.spread_points = 16.0
        payload.spread_atr_reference_points = 20.0
        payload.low_equity_monte_carlo_floor = 0.88
        payload.candidate_monte_carlo_win_rate = 0.80
        payload.trade_quality_score = 0.55
        payload.execution_quality_score = 0.59
        payload.execution_minute_quality_score = 0.58
        payload.session_quality_score = 0.54
        payload.microstructure_composite_score = 0.45
        payload.lead_lag_alignment_score = -0.12
        payload.setup = "XAUUSD_M5_GRID_SCALPER_START"
        payload.news_state = "NEWS_CAUTION"

        decision = self.engine.evaluate(payload)

        self.assertFalse(decision.approved)
        self.assertIn(decision.reason, {"low_equity_mc_floor", "daily_normal_quality_block"})

    def test_low_equity_spread_guard_blocks_disorder(self) -> None:
        payload = self._base()
        payload.equity = 90.0
        payload.account_balance = 90.0
        payload.margin_free = 90.0
        payload.requested_risk_pct = 0.02
        payload.hard_risk_cap = 0.02
        payload.spread_points = 28.0
        payload.spread_atr_reference_points = 20.0
        payload.candidate_monte_carlo_win_rate = 0.92

        decision = self.engine.evaluate(payload)

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "low_equity_spread_atr_guard")

    def test_low_equity_attack_override_bypasses_spread_guard_for_strong_usdjpy_candidate(self) -> None:
        payload = self._base()
        payload.symbol = "USDJPY"
        payload.setup = "USDJPY_SESSION_PULLBACK"
        payload.current_time = datetime(2026, 3, 27, 8, 0, tzinfo=timezone.utc)
        payload.equity = 26.76
        payload.account_balance = 22.65
        payload.margin_free = 26.76
        payload.requested_risk_pct = 0.01
        payload.hard_risk_cap = 0.02
        payload.spread_points = 27.0
        payload.spread_atr_reference_points = 20.0
        payload.candidate_monte_carlo_win_rate = 0.81
        payload.trade_quality_score = 0.84
        payload.execution_quality_score = 0.82
        payload.execution_minute_quality_score = 0.76
        payload.session_quality_score = 0.74
        payload.expected_value_r = 0.96
        payload.confluence_score = 4.12
        payload.session_native_pair = True
        payload.session_priority_multiplier = 1.2
        payload.lead_lag_alignment_score = 0.06
        payload.microstructure_composite_score = 0.44
        payload.news_state = "NEWS_CAUTION"

        decision = self.engine.evaluate(payload)

        self.assertNotEqual(decision.reason, "low_equity_spread_atr_guard")

    def test_low_equity_attack_override_bypasses_spread_guard_for_strong_nas100_candidate(self) -> None:
        payload = self._base()
        payload.symbol = "NAS100"
        payload.setup = "NAS100_VWAP_PULLBACK"
        payload.entry_price = 21540.0
        payload.stop_price = 21505.0
        payload.tp_price = 21610.0
        payload.current_time = datetime(2026, 3, 27, 8, 0, tzinfo=timezone.utc)
        payload.equity = 26.76
        payload.account_balance = 22.65
        payload.margin_free = 26.76
        payload.requested_risk_pct = 0.01
        payload.hard_risk_cap = 0.02
        payload.spread_points = 25.0
        payload.spread_atr_reference_points = 18.0
        payload.candidate_monte_carlo_win_rate = 0.79
        payload.trade_quality_score = 0.82
        payload.execution_quality_score = 0.78
        payload.execution_minute_quality_score = 0.74
        payload.session_quality_score = 0.71
        payload.expected_value_r = 0.92
        payload.confluence_score = 4.20
        payload.session_native_pair = True
        payload.session_priority_multiplier = 1.18
        payload.lead_lag_alignment_score = 0.08
        payload.microstructure_composite_score = 0.48
        payload.news_state = "NEWS_CAUTION"

        decision = self.engine.evaluate(payload)

        self.assertNotEqual(decision.reason, "low_equity_spread_atr_guard")

    def test_low_equity_attack_override_bypasses_hard_spread_cap_for_strong_nas100_candidate(self) -> None:
        payload = self._base()
        payload.symbol = "NAS100"
        payload.setup = "NAS100_OPENING_DRIVE_BREAKOUT"
        payload.entry_price = 24053.15
        payload.stop_price = 23992.67
        payload.tp_price = 24174.11
        payload.current_time = datetime(2026, 3, 27, 9, 10, tzinfo=timezone.utc)
        payload.equity = 26.76
        payload.account_balance = 22.65
        payload.margin_free = 26.76
        payload.requested_risk_pct = 0.01
        payload.hard_risk_cap = 0.02
        payload.max_spread_points = 35.0
        payload.spread_points = 80.0
        payload.spread_atr_reference_points = 18.0
        payload.candidate_monte_carlo_win_rate = 0.79
        payload.trade_quality_score = 0.80
        payload.execution_quality_score = 0.72
        payload.execution_minute_quality_score = 0.68
        payload.session_quality_score = 0.70
        payload.ai_probability = 0.80
        payload.expected_value_r = 0.92
        payload.confluence_score = 4.0
        payload.session_native_pair = True
        payload.session_priority_multiplier = 1.18
        payload.lead_lag_alignment_score = 0.08
        payload.microstructure_composite_score = 0.48
        payload.news_state = "NEWS_CAUTION"

        decision = self.engine.evaluate(payload)

        self.assertNotEqual(decision.reason, "spread_too_wide")

    def test_low_equity_attack_override_bypasses_mc_floor_for_strong_nas100_candidate(self) -> None:
        payload = self._base()
        payload.symbol = "NAS100"
        payload.setup = "NAS100_OPENING_DRIVE_BREAKOUT"
        payload.entry_price = 24053.15
        payload.stop_price = 23992.67
        payload.tp_price = 24174.11
        payload.current_time = datetime(2026, 3, 27, 9, 10, tzinfo=timezone.utc)
        payload.equity = 26.76
        payload.account_balance = 22.65
        payload.margin_free = 26.76
        payload.requested_risk_pct = 0.01
        payload.hard_risk_cap = 0.02
        payload.max_spread_points = 35.0
        payload.spread_points = 80.0
        payload.spread_atr_reference_points = 18.0
        payload.candidate_monte_carlo_win_rate = 0.79
        payload.trade_quality_score = 0.80
        payload.execution_quality_score = 0.72
        payload.execution_minute_quality_score = 0.68
        payload.session_quality_score = 0.70
        payload.ai_probability = 0.80
        payload.expected_value_r = 0.92
        payload.confluence_score = 4.0
        payload.session_native_pair = True
        payload.session_priority_multiplier = 1.18
        payload.lead_lag_alignment_score = 0.08
        payload.microstructure_composite_score = 0.48
        payload.news_state = "NEWS_CAUTION"

        decision = self.engine.evaluate(payload)

        self.assertNotEqual(decision.reason, "low_equity_mc_floor")

    def test_blocks_weekend_when_symbol_not_always_on(self) -> None:
        payload = self._base()
        payload.current_time = datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc)
        decision = self.engine.evaluate(payload)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "weekend_disabled")

    def test_allows_weekend_for_always_on_symbol(self) -> None:
        payload = self._base()
        payload.symbol = "BTCUSD"
        payload.current_time = datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc)
        payload.weekend_trading_allowed = True
        decision = self.engine.evaluate(payload)
        self.assertTrue(decision.approved)

    def test_allows_forex_after_sunday_new_york_open(self) -> None:
        payload = self._base()
        payload.symbol = "EURUSD"
        payload.current_time = datetime(2026, 3, 8, 23, 15, tzinfo=timezone.utc)
        decision = self.engine.evaluate(payload)
        self.assertTrue(decision.approved)

    def test_funded_detector_stays_off_for_normal_retail_descriptor(self) -> None:
        self.assertFalse(
            detect_funded_account_mode("Main", "ICMarketsSC-Live09", "Raw Trading Ltd", "IC Markets")
        )

    def test_funded_detector_enables_for_known_provider_descriptor(self) -> None:
        self.assertTrue(
            detect_funded_account_mode("FTMO Challenge", "FTMO-Server", "FTMO", "FTMO")
        )

    def test_funded_detector_requires_more_than_generic_prop_word(self) -> None:
        self.assertFalse(
            detect_funded_account_mode("Main Prop Research", "Live01", "Private Broker", "Broker Name")
        )

    def test_micro_mode_risk_ramps_with_closed_trade_count(self) -> None:
        low = self._base()
        low.micro_enabled = True
        low.stats.closed_trades_total = 0
        low.stats.trades_today = 5
        low.micro_max_loss_usd = 25.0
        low.micro_total_risk_usd = 50.0
        low_decision = self.engine.evaluate(low)

        high = self._base()
        high.micro_enabled = True
        high.stats.closed_trades_total = 50
        high.stats.trades_today = 5
        high.micro_max_loss_usd = 25.0
        high.micro_total_risk_usd = 50.0
        high_decision = self.engine.evaluate(high)

        self.assertTrue(low_decision.approved)
        self.assertTrue(high_decision.approved)
        self.assertLess(low_decision.risk_pct, high_decision.risk_pct)

    def test_first_trade_protection_reduces_size(self) -> None:
        early = self._base()
        early.micro_enabled = True
        early.stats.trades_today = 0
        early.stats.closed_trades_total = 50
        early.micro_max_loss_usd = 25.0
        early.micro_total_risk_usd = 50.0
        early_decision = self.engine.evaluate(early)

        later = self._base()
        later.micro_enabled = True
        later.stats.trades_today = 3
        later.stats.closed_trades_total = 50
        later.micro_max_loss_usd = 25.0
        later.micro_total_risk_usd = 50.0
        later_decision = self.engine.evaluate(later)

        self.assertTrue(early_decision.approved)
        self.assertTrue(later_decision.approved)
        self.assertLess(early_decision.volume, later_decision.volume)

    def test_fixed_lot_rejected_when_implied_risk_exceeds_budget(self) -> None:
        payload = self._base()
        payload.use_fixed_lot = True
        payload.fixed_lot = 1.0
        payload.stats.trades_today = 5
        decision = self.engine.evaluate(payload)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "fixed_lot_risk_exceeds_budget")

    def test_stop_too_tight_is_rejected(self) -> None:
        payload = self._base()
        payload.entry_price = 2200.0
        payload.stop_price = 2199.95
        payload.min_stop_distance_points = 10.0
        payload.symbol_point = 0.01
        decision = self.engine.evaluate(payload)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "stop_too_tight")

    def test_micro_min_lot_trade_is_blocked_when_usd_loss_exceeds_cap(self) -> None:
        payload = self._base()
        payload.micro_enabled = True
        payload.stats.trades_today = 5
        payload.requested_risk_pct = 0.0001
        payload.entry_price = 2200.0
        payload.stop_price = 2195.0
        payload.volume_min = 0.01
        payload.micro_max_loss_usd = 2.5
        payload.contract_size = 100.0
        decision = self.engine.evaluate(payload)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "micro_survival_trade_risk_exceeds_usd")

    def test_micro_total_open_risk_cap_blocks_new_trade(self) -> None:
        payload = self._base()
        payload.micro_enabled = True
        payload.stats.trades_today = 5
        payload.entry_price = 2200.0
        payload.stop_price = 2198.0
        payload.volume_min = 0.01
        payload.contract_size = 100.0
        payload.projected_open_risk_usd = 4.5
        payload.micro_total_risk_usd = 5.0
        decision = self.engine.evaluate(payload)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "micro_survival_total_risk_exceeds_usd")

    def test_loss_streak_cooldown_blocks_same_day_after_threshold_losses(self) -> None:
        payload = self._base()
        payload.stats = TradeStats(consecutive_losses=7, cooldown_trades_remaining=2)

        decision = self.engine.evaluate(payload)

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "loss_streak_cooldown")

    def test_btc_bootstrap_fixed_min_lot_is_executable_with_live_tick_economics(self) -> None:
        payload = self._base()
        payload.symbol = "BTCUSD"
        payload.current_time = datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc)
        payload.weekend_trading_allowed = True
        payload.entry_price = 68000.25
        payload.stop_price = 67998.75
        payload.tp_price = 68002.12
        payload.equity = 50.0
        payload.account_balance = 50.0
        payload.margin_free = 50.0
        payload.micro_enabled = True
        payload.use_fixed_lot = True
        payload.fixed_lot = 0.01
        payload.volume_min = 0.01
        payload.volume_step = 0.01
        payload.volume_max = 10.0
        payload.contract_size = 1.0
        payload.symbol_point = 0.01
        payload.symbol_tick_size = 0.01
        payload.symbol_tick_value = 1.0
        payload.micro_risk_pct_ceiling = 0.02
        payload.bootstrap_enabled = True
        payload.bootstrap_equity_threshold = 160.0
        payload.bootstrap_per_trade_hard_cap = 4.0
        payload.bootstrap_total_exposure_cap = 10.0
        payload.bootstrap_min_risk_amount = 1.0
        payload.bootstrap_min_lot_risk_multiplier = 6.0
        payload.stats.trades_today = 0
        payload.stats.closed_trades_total = 0

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertIn("bootstrap", str(decision.reason))

    def test_soft_daily_dd_blocks_non_elite_setups(self) -> None:
        payload = self._base()
        payload.stats.daily_dd_pct_live = 0.031
        payload.ai_probability = 0.60
        payload.confluence_score = 3.0
        payload.expected_value_r = 0.20

        decision = self.engine.evaluate(payload)

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "daily_caution_quality_block")

    def test_funded_mode_blocks_when_drawdown_buffer_is_effectively_exhausted(self) -> None:
        payload = self._base()
        payload.funded_account_mode = True
        payload.funded_daily_loss_limit_pct = 0.05
        payload.funded_overall_drawdown_limit_pct = 0.10
        payload.funded_guard_buffer_pct = 0.02
        payload.stats.daily_dd_pct_live = 0.0495
        payload.stats.absolute_drawdown_pct = 0.04

        decision = self.engine.evaluate(payload)

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "funded_buffer_exhausted")

    def test_soft_daily_dd_allows_elite_setups(self) -> None:
        payload = self._base()
        payload.stats.daily_dd_pct_live = 0.031
        payload.ai_probability = 0.80
        payload.confluence_score = 4.2
        payload.expected_value_r = 0.50
        payload.current_phase = "PHASE_2"
        payload.trade_quality_score = 0.90
        payload.regime_confidence = 0.78
        payload.execution_quality_state = "GOOD"

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertTrue(bool(decision.diagnostics.get("soft_dd_active")))

    def test_hard_daily_dd_blocks_new_entries(self) -> None:
        payload = self._base()
        payload.stats.daily_dd_pct_live = 0.071

        decision = self.engine.evaluate(payload)

        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "hard_daily_dd")

    def test_recovery_mode_allows_defensive_reentry_after_hard_daily_dd(self) -> None:
        payload = self._base()
        payload.stats.daily_dd_pct_live = 0.071
        payload.recovery_mode_active = True
        payload.trade_quality_score = 0.92
        payload.trade_quality_band = "A"
        payload.ai_probability = 0.80
        payload.expected_value_r = 0.55
        payload.confluence_score = 4.5
        payload.session_native_pair = True
        payload.session_priority_multiplier = 1.12

        decision = self.engine.evaluate(payload)

        self.assertNotEqual(decision.reason, "hard_daily_dd")
        self.assertNotEqual(decision.kill, "HARD")

    def test_daily_caution_timeout_releases_after_four_hours(self) -> None:
        payload = self._base()
        payload.stats.daily_dd_pct_live = 0.031
        payload.stats.trading_day_key = "2026-03-05"
        payload.trade_quality_band = "B"
        payload.trade_quality_score = 0.62
        payload.ai_probability = 0.62
        payload.expected_value_r = 0.32
        payload.confluence_score = 3.2
        payload.daily_governor_started_at = datetime(2026, 3, 4, 8, 30, tzinfo=timezone.utc).isoformat()
        payload.daily_governor_trigger_day_key = "2026-03-05"
        payload.daily_governor_timeout_hours = 4.0

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)

    def test_daily_hard_stop_timeout_degrades_to_defensive_flow(self) -> None:
        payload = self._base()
        payload.stats.daily_dd_pct_live = 0.071
        payload.stats.daily_pnl_pct = -0.02
        payload.stats.trading_day_key = "2026-03-05"
        payload.daily_governor_started_at = datetime(2026, 3, 4, 8, 30, tzinfo=timezone.utc).isoformat()
        payload.daily_governor_trigger_day_key = "2026-03-05"
        payload.daily_governor_timeout_hours = 4.0
        payload.trade_quality_score = 0.92
        payload.trade_quality_band = "A"
        payload.ai_probability = 0.80
        payload.expected_value_r = 0.55
        payload.confluence_score = 4.5
        payload.session_native_pair = True
        payload.session_priority_multiplier = 1.12

        decision = self.engine.evaluate(payload)

        self.assertNotEqual(decision.reason, "hard_daily_dd")
        self.assertNotEqual(decision.kill, "HARD")

    def test_daily_governor_releases_on_new_sydney_day(self) -> None:
        payload = self._base()
        payload.stats.daily_dd_pct_live = 0.031
        payload.stats.trading_day_key = "2026-03-05"
        payload.daily_governor_trigger_day_key = "2026-03-04"
        payload.trade_quality_band = "B"
        payload.trade_quality_score = 0.62
        payload.ai_probability = 0.62
        payload.expected_value_r = 0.32
        payload.confluence_score = 3.2

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)

    def test_same_lane_loss_streak_triggers_caution_mode(self) -> None:
        payload = self._base()
        payload.lane_consecutive_losses = 3
        payload.same_lane_loss_caution_streak = 3
        payload.trade_quality_score = 0.82
        payload.trade_quality_band = "A"
        payload.ai_probability = 0.80
        payload.expected_value_r = 0.45
        payload.confluence_score = 4.0

        governor = self.engine._resolve_daily_governor(payload, spread_elevated=False)  # noqa: SLF001

        self.assertEqual(governor.state, "DAILY_CAUTION")
        self.assertEqual(governor.reason, "same_lane_loss_streak_caution")

    def test_daily_caution_preserves_strong_native_session_lane_before_non_native(self) -> None:
        native = self._base()
        native.symbol = "AUDJPY"
        native.stats.daily_dd_pct_live = 0.031
        native.trade_quality_band = "B"
        native.trade_quality_score = 0.62
        native.ai_probability = 0.62
        native.expected_value_r = 0.32
        native.confluence_score = 3.2
        native.session_native_pair = True
        native.session_priority_multiplier = 1.14
        native.lane_strength_multiplier = 1.08
        native.lane_budget_share = 0.40
        native.strategy_family = "TREND"

        non_native = self._base()
        non_native.symbol = "EURUSD"
        non_native.stats.daily_dd_pct_live = 0.031
        non_native.trade_quality_band = "B"
        non_native.trade_quality_score = 0.62
        non_native.ai_probability = 0.62
        non_native.expected_value_r = 0.32
        non_native.confluence_score = 3.2
        non_native.session_native_pair = False
        non_native.session_priority_multiplier = 0.92
        non_native.lane_strength_multiplier = 0.95
        non_native.lane_budget_share = 0.10
        non_native.strategy_family = "TREND"

        native_decision = self.engine.evaluate(native)
        non_native_decision = self.engine.evaluate(non_native)

        self.assertTrue(native_decision.approved, msg=native_decision.reason)
        self.assertFalse(non_native_decision.approved)
        self.assertEqual(non_native_decision.reason, "daily_caution_quality_block")

    def test_daily_defensive_preserves_strong_native_session_lane_before_non_native(self) -> None:
        native = self._base()
        native.symbol = "AUDJPY"
        native.stats.daily_dd_pct_live = 0.055
        native.trade_quality_band = "B+"
        native.trade_quality_score = 0.69
        native.ai_probability = 0.69
        native.expected_value_r = 0.36
        native.confluence_score = 3.8
        native.session_native_pair = True
        native.session_priority_multiplier = 1.14
        native.lane_strength_multiplier = 1.08
        native.lane_budget_share = 0.40
        native.strategy_family = "TREND"

        non_native = self._base()
        non_native.symbol = "EURUSD"
        non_native.stats.daily_dd_pct_live = 0.055
        non_native.trade_quality_band = "B+"
        non_native.trade_quality_score = 0.69
        non_native.ai_probability = 0.69
        non_native.expected_value_r = 0.36
        non_native.confluence_score = 3.8
        non_native.session_native_pair = False
        non_native.session_priority_multiplier = 0.92
        non_native.lane_strength_multiplier = 0.95
        non_native.lane_budget_share = 0.10
        non_native.strategy_family = "TREND"

        native_decision = self.engine.evaluate(native)
        non_native_decision = self.engine.evaluate(non_native)

        self.assertTrue(native_decision.approved, msg=native_decision.reason)
        self.assertFalse(non_native_decision.approved)
        self.assertEqual(non_native_decision.reason, "daily_defensive_quality_block")

    def test_lane_expectancy_allocator_shifts_more_capacity_to_top_lane(self) -> None:
        strong = self._base()
        strong.max_trades_per_day = 20
        strong.max_trades_per_hour = 4
        strong.trade_quality_score = 0.74
        strong.ai_probability = 0.72
        strong.expected_value_r = 0.34
        strong.confluence_score = 3.8
        strong.lane_budget_share = 0.20
        strong.lane_expectancy_multiplier = 1.18
        strong.lane_expectancy_score = 0.72

        weak = self._base()
        weak.max_trades_per_day = 20
        weak.max_trades_per_hour = 4
        weak.trade_quality_score = 0.74
        weak.ai_probability = 0.72
        weak.expected_value_r = 0.34
        weak.confluence_score = 3.8
        weak.lane_budget_share = 0.20
        weak.lane_expectancy_multiplier = 0.90
        weak.lane_expectancy_score = 0.34

        strong_decision = self.engine.evaluate(strong)
        weak_decision = self.engine.evaluate(weak)

        self.assertTrue(strong_decision.approved, msg=strong_decision.reason)
        self.assertTrue(weak_decision.approved, msg=weak_decision.reason)
        self.assertGreater(
            int(strong_decision.diagnostics.get("projected_trade_capacity_today") or 0),
            int(weak_decision.diagnostics.get("projected_trade_capacity_today") or 0),
        )
        self.assertGreater(
            int(strong_decision.diagnostics.get("lane_available_capacity") or 0),
            int(weak_decision.diagnostics.get("lane_available_capacity") or 0),
        )

    def test_bootstrap_first_trade_sl_width_can_pass_wider_fx_stop_in_small_account_mode(self) -> None:
        payload = self._base()
        payload.symbol = "EURUSD"
        payload.micro_enabled = True
        payload.equity = 57.86
        payload.account_balance = 57.86
        payload.margin_free = 57.86
        payload.stats.trades_today = 0
        payload.stats.closed_trades_total = 5
        payload.symbol_point = 0.00001
        payload.symbol_tick_size = 0.00001
        payload.symbol_digits = 5
        payload.atr_current = 0.00025
        payload.entry_price = 1.08420
        payload.stop_price = 1.08349
        payload.tp_price = 1.08540
        payload.contract_size = 100000.0
        payload.volume_min = 0.01
        payload.volume_step = 0.01
        payload.bootstrap_enabled = True
        payload.bootstrap_equity_threshold = 160.0
        payload.bootstrap_first_trade_max_sl_atr = 3.0

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)

    def test_bootstrap_mode_transitions_back_to_standard_above_threshold(self) -> None:
        payload = self._base()
        payload.symbol = "BTCUSD"
        payload.equity = 250.0
        payload.account_balance = 250.0
        payload.margin_free = 250.0
        payload.micro_enabled = True
        payload.bootstrap_enabled = True
        payload.bootstrap_equity_threshold = 160.0
        payload.micro_max_loss_usd = 25.0
        payload.micro_total_risk_usd = 50.0

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertEqual(str(decision.diagnostics.get("risk_mode")), "standard")

    def test_undertrading_boost_survives_bootstrap_mode(self) -> None:
        base = self._base()
        base.symbol = "BTCUSD"
        base.current_time = datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc)
        base.weekend_trading_allowed = True
        base.equity = 50.0
        base.account_balance = 50.0
        base.margin_free = 50.0
        base.micro_enabled = True
        base.bootstrap_enabled = True
        base.bootstrap_equity_threshold = 160.0
        base.bootstrap_per_trade_hard_cap = 4.0
        base.bootstrap_total_exposure_cap = 10.0
        base.bootstrap_min_risk_amount = 1.0
        base.micro_risk_pct_ceiling = 0.02
        base.use_fixed_lot = False

        boosted = self._base()
        boosted.symbol = "BTCUSD"
        boosted.current_time = datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc)
        boosted.weekend_trading_allowed = True
        boosted.equity = 50.0
        boosted.account_balance = 50.0
        boosted.margin_free = 50.0
        boosted.micro_enabled = True
        boosted.bootstrap_enabled = True
        boosted.bootstrap_equity_threshold = 160.0
        boosted.bootstrap_per_trade_hard_cap = 4.0
        boosted.bootstrap_total_exposure_cap = 10.0
        boosted.bootstrap_min_risk_amount = 1.0
        boosted.micro_risk_pct_ceiling = 0.02
        boosted.no_trade_boost_enabled = True
        boosted.no_trade_boost_eligible = True
        boosted.no_trade_boost_elapsed_minutes = 120.0
        boosted.no_trade_boost_after_minutes = 30
        boosted.no_trade_boost_interval_minutes = 15
        boosted.no_trade_boost_step_pct = 0.01
        boosted.no_trade_boost_max_pct = 0.08

        base_decision = self.engine.evaluate(base)
        boosted_decision = self.engine.evaluate(boosted)

        self.assertTrue(base_decision.approved, msg=base_decision.reason)
        self.assertTrue(boosted_decision.approved, msg=boosted_decision.reason)
        self.assertGreater(boosted_decision.risk_pct, base_decision.risk_pct)

    def test_bootstrap_drawdown_kill_is_more_permissive_than_standard(self) -> None:
        payload = self._base()
        payload.symbol = "BTCUSD"
        payload.current_time = datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc)
        payload.weekend_trading_allowed = True
        payload.equity = 63.57
        payload.account_balance = 53.57
        payload.margin_free = 63.57
        payload.micro_enabled = True
        payload.bootstrap_enabled = True
        payload.bootstrap_equity_threshold = 160.0
        payload.bootstrap_drawdown_kill = 0.12
        payload.stats.rolling_drawdown_pct = 0.055

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)

    def test_bootstrap_absolute_drawdown_hard_stop_uses_bootstrap_threshold(self) -> None:
        payload = self._base()
        payload.symbol = "BTCUSD"
        payload.current_time = datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc)
        payload.weekend_trading_allowed = True
        payload.equity = 63.57
        payload.account_balance = 53.57
        payload.margin_free = 63.57
        payload.micro_enabled = True
        payload.bootstrap_enabled = True
        payload.bootstrap_equity_threshold = 160.0
        payload.bootstrap_drawdown_kill = 0.12
        payload.stats.absolute_drawdown_pct = 0.089

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)

    def test_xau_grid_first_trade_allows_atr_wider_stop_when_candidate_requires_it(self) -> None:
        payload = self._base()
        payload.micro_enabled = True
        payload.stats.trades_today = 0
        payload.stats.closed_trades_total = 0
        payload.atr_current = 1.0
        payload.equity = 50.0
        payload.account_balance = 50.0
        payload.margin_free = 200.0
        payload.entry_price = 2200.0
        payload.stop_price = 2197.5  # 2.5 ATR
        payload.setup = "XAUUSD_M5_GRID_SCALPER_START"
        payload.candidate_stop_atr = 2.5
        payload.requested_risk_pct = 0.10
        payload.strategy_risk_cap = 0.10
        payload.skip_micro_risk_clamp = True
        payload.symbol_tick_size = 0.01
        payload.symbol_tick_value = 1.0
        payload.micro_max_loss_usd = 25.0
        payload.micro_total_risk_usd = 50.0

        decision = self.engine.evaluate(payload)
        self.assertTrue(decision.approved, msg=decision.reason)

    def test_first_trade_sl_width_uses_points_allows_within_limit_for_xau(self) -> None:
        payload = self._base()
        payload.micro_enabled = True
        payload.stats.trades_today = 0
        payload.stats.closed_trades_total = 0
        payload.atr_current = 0.5
        payload.entry_price = 2200.0
        payload.stop_price = 2199.2  # 0.8 price -> 80 points @ 0.01
        payload.first_trade_max_sl_atr = 2.0  # 1.0 price -> 100 points
        payload.symbol_point = 0.01
        payload.micro_max_loss_usd = 100.0
        payload.micro_total_risk_usd = 200.0

        decision = self.engine.evaluate(payload)
        self.assertTrue(decision.approved, msg=decision.reason)

    def test_first_trade_sl_width_uses_points_blocks_above_limit_for_xau(self) -> None:
        payload = self._base()
        payload.micro_enabled = True
        payload.stats.trades_today = 0
        payload.stats.closed_trades_total = 0
        payload.atr_current = 0.5
        payload.entry_price = 2200.0
        payload.stop_price = 2198.8  # 1.2 price -> 120 points @ 0.01
        payload.first_trade_max_sl_atr = 2.0  # 1.0 price -> 100 points
        payload.symbol_point = 0.01
        payload.micro_max_loss_usd = 100.0
        payload.micro_total_risk_usd = 200.0

        decision = self.engine.evaluate(payload)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "first_trade_sl_too_wide")

    def test_bootstrap_high_min_lot_symbol_can_use_bounded_tolerance(self) -> None:
        payload = self._base()
        payload.symbol = "USOIL"
        payload.mode = "LIVE"
        payload.live_enabled = True
        payload.live_allowed = True
        payload.micro_enabled = True
        payload.bootstrap_enabled = True
        payload.bootstrap_equity_threshold = 160.0
        payload.bootstrap_per_trade_hard_cap = 4.0
        payload.bootstrap_total_exposure_cap = 10.0
        payload.bootstrap_min_risk_amount = 1.0
        payload.equity = 82.86
        payload.account_balance = 72.0
        payload.margin_free = 82.86
        payload.stats.trades_today = 0
        payload.stats.closed_trades_total = 0
        payload.entry_price = 77.52
        payload.stop_price = 74.84999938964844
        payload.tp_price = 81.80
        payload.symbol_point = 0.01
        payload.symbol_tick_size = 0.01
        payload.symbol_tick_value = 0.017057
        payload.contract_size = 1.0
        payload.volume_min = 1.0
        payload.volume_step = 1.0
        payload.volume_max = 10.0
        payload.atr_current = 0.846
        payload.atr_average = 0.70
        payload.max_spread_points = 60.0
        payload.spread_points = 24.0
        payload.current_time = datetime(2026, 3, 9, 1, 37, tzinfo=timezone.utc)

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertEqual(decision.volume, 1.0)
        self.assertEqual(str(decision.diagnostics.get("risk_mode")), "bootstrap")

    def test_bootstrap_balanced_index_min_lot_can_use_bounded_tolerance(self) -> None:
        payload = self._base()
        payload.symbol = "NAS100"
        payload.mode = "LIVE"
        payload.live_enabled = True
        payload.live_allowed = True
        payload.micro_enabled = True
        payload.bootstrap_enabled = True
        payload.bootstrap_equity_threshold = 160.0
        payload.bootstrap_per_trade_hard_cap = 4.0
        payload.bootstrap_total_exposure_cap = 10.0
        payload.bootstrap_min_risk_amount = 1.0
        payload.equity = 85.0
        payload.account_balance = 70.0
        payload.margin_free = 80.0
        payload.stats.trades_today = 3
        payload.stats.closed_trades_total = 10
        payload.entry_price = 24005.0
        payload.stop_price = 23938.84
        payload.tp_price = 24124.0
        payload.symbol_point = 0.01
        payload.symbol_tick_size = 0.01
        payload.symbol_tick_value = 0.01
        payload.contract_size = 1.0
        payload.volume_min = 0.1
        payload.volume_step = 0.1
        payload.volume_max = 500.0
        payload.max_spread_points = 120.0
        payload.spread_points = 80.0
        payload.current_time = datetime(2026, 3, 9, 2, 18, tzinfo=timezone.utc)

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertEqual(decision.volume, 0.1)
        self.assertEqual(decision.reason, "approved_bootstrap_min_lot")

    def test_first_trade_sl_width_regression_prevents_points_price_mismatch_false_block(self) -> None:
        payload = self._base()
        payload.micro_enabled = True
        payload.stats.trades_today = 0
        payload.stats.closed_trades_total = 0
        payload.atr_current = 0.5
        payload.entry_price = 2200.0
        payload.stop_price = 2199.2  # should be allowed by 100-point limit
        payload.first_trade_max_sl_atr = 2.0
        payload.symbol_point = 0.01
        payload.micro_max_loss_usd = 100.0
        payload.micro_total_risk_usd = 200.0

        decision = self.engine.evaluate(payload)
        self.assertNotEqual(decision.reason, "first_trade_sl_too_wide")
        self.assertTrue(decision.approved, msg=decision.reason)

    def test_bootstrap_first_trade_sl_width_allows_reasonable_wider_stop(self) -> None:
        payload = self._base()
        payload.micro_enabled = True
        payload.bootstrap_enabled = True
        payload.bootstrap_equity_threshold = 160.0
        payload.bootstrap_first_trade_max_sl_atr = 2.2
        payload.stats.trades_today = 0
        payload.stats.closed_trades_total = 0
        payload.equity = 57.86
        payload.account_balance = 47.86
        payload.margin_free = 57.86
        payload.atr_current = 0.5
        payload.entry_price = 1.15308
        payload.stop_price = 1.15237
        payload.first_trade_max_sl_atr = 1.4
        payload.symbol_point = 0.00001
        payload.symbol_tick_size = 0.00001
        payload.symbol_tick_value = 1.707825
        payload.volume_min = 0.01
        payload.volume_step = 0.01
        payload.micro_max_loss_usd = 2.5
        payload.micro_total_risk_usd = 5.0
        payload.bootstrap_per_trade_hard_cap = 4.0
        payload.bootstrap_total_exposure_cap = 10.0

        decision = self.engine.evaluate(payload)
        self.assertTrue(decision.approved, msg=decision.reason)

    def test_first_trade_sl_width_uses_tick_size_when_point_missing(self) -> None:
        payload = self._base()
        payload.micro_enabled = True
        payload.stats.trades_today = 0
        payload.stats.closed_trades_total = 0
        payload.atr_current = 0.5
        payload.entry_price = 2200.0
        payload.stop_price = 2198.8
        payload.first_trade_max_sl_atr = 2.0
        payload.symbol_point = 0.0
        payload.symbol_tick_size = 0.01
        payload.symbol_digits = 2
        payload.micro_max_loss_usd = 100.0
        payload.micro_total_risk_usd = 200.0

        decision = self.engine.evaluate(payload)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "first_trade_sl_too_wide")
        self.assertEqual(str(decision.diagnostics["point_source"]), "symbol_tick_size")

    def test_xau_grid_cycle_budget_10pct_allows_min_lot_while_2pct_blocks(self) -> None:
        low = self._base()
        low.symbol = "XAUUSD"
        low.setup = "XAUUSD_M5_GRID_SCALPER_START"
        low.equity = 50.0
        low.margin_free = 200.0
        low.entry_price = 2200.0
        low.stop_price = 2197.5
        low.symbol_tick_size = 0.01
        low.symbol_tick_value = 1.0
        low.volume_min = 0.01
        low.volume_step = 0.01
        low.requested_risk_pct = 0.02
        low.hard_risk_cap = 0.005
        low.strategy_risk_cap = 0.10
        low.skip_micro_risk_clamp = True
        low.micro_enabled = False

        high = self._base()
        high.symbol = "XAUUSD"
        high.setup = "XAUUSD_M5_GRID_SCALPER_START"
        high.equity = 50.0
        high.margin_free = 200.0
        high.entry_price = 2200.0
        high.stop_price = 2197.5
        high.symbol_tick_size = 0.01
        high.symbol_tick_value = 1.0
        high.volume_min = 0.01
        high.volume_step = 0.01
        high.requested_risk_pct = 0.10
        high.hard_risk_cap = 0.005
        high.strategy_risk_cap = 0.10
        high.skip_micro_risk_clamp = True
        high.micro_enabled = False

        low_decision = self.engine.evaluate(low)
        high_decision = self.engine.evaluate(high)
        self.assertFalse(low_decision.approved)
        self.assertEqual(low_decision.reason, "lot_below_min_or_margin_too_low")
        self.assertTrue(high_decision.approved, msg=high_decision.reason)
        self.assertGreaterEqual(high_decision.volume, 0.01)

    def test_xau_grid_strong_bootstrap_min_lot_override_survives_budget_floor(self) -> None:
        payload = self._base()
        payload.symbol = "XAUUSD"
        payload.setup = "XAUUSD_M5_GRID_SCALPER_START"
        payload.current_time = datetime(2026, 3, 27, 9, 8, tzinfo=timezone.utc)
        payload.equity = 26.76
        payload.account_balance = 22.65
        payload.margin_free = 26.76
        payload.entry_price = 4469.10
        payload.stop_price = 4467.34
        payload.tp_price = 4472.20
        payload.symbol_tick_size = 0.01
        payload.symbol_tick_value = 1.0
        payload.volume_min = 0.01
        payload.volume_step = 0.01
        payload.volume_max = 5.0
        payload.max_loss_usd_floor = 0.5
        payload.requested_risk_pct = 0.08
        payload.hard_risk_cap = 0.03
        payload.strategy_risk_cap = 0.08
        payload.skip_micro_risk_clamp = True
        payload.micro_enabled = False
        payload.spread_points = 15.0
        payload.max_spread_points = 35.0
        payload.spread_atr_reference_points = 9.0
        payload.trade_quality_score = 0.86
        payload.execution_quality_score = 0.92
        payload.execution_minute_quality_score = 0.83
        payload.session_quality_score = 0.81
        payload.ai_probability = 0.82
        payload.expected_value_r = 1.64
        payload.confluence_score = 5.0
        payload.candidate_monte_carlo_win_rate = 0.80
        payload.session_native_pair = True
        payload.session_priority_multiplier = 1.2
        payload.projected_cycle_risk_usd = 1.62

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertGreaterEqual(decision.volume, 0.01)

    def test_no_trade_boost_increases_effective_risk_and_caps(self) -> None:
        payload = self._base()
        payload.use_fixed_lot = True
        payload.fixed_lot = 0.01
        payload.stats.trades_today = 5
        payload.requested_risk_pct = 0.02
        payload.hard_risk_cap = 0.20
        payload.no_trade_boost_enabled = True
        payload.no_trade_boost_eligible = True
        payload.no_trade_boost_after_minutes = 60
        payload.no_trade_boost_interval_minutes = 15
        payload.no_trade_boost_step_pct = 0.02
        payload.no_trade_boost_max_pct = 0.10
        payload.no_trade_boost_elapsed_minutes = 90
        boosted = self.engine.evaluate(payload)
        self.assertTrue(boosted.approved, msg=boosted.reason)
        self.assertAlmostEqual(float(boosted.risk_pct), 0.08, places=6)
        self.assertTrue(bool(boosted.diagnostics.get("risk_boost_active", False)))

        payload.no_trade_boost_elapsed_minutes = 600
        capped = self.engine.evaluate(payload)
        self.assertTrue(capped.approved, msg=capped.reason)
        self.assertAlmostEqual(float(capped.risk_pct), 0.10, places=6)

    def test_no_trade_boost_not_applied_when_ineligible(self) -> None:
        payload = self._base()
        payload.use_fixed_lot = True
        payload.fixed_lot = 0.01
        payload.stats.trades_today = 5
        payload.requested_risk_pct = 0.02
        payload.hard_risk_cap = 0.20
        payload.no_trade_boost_enabled = True
        payload.no_trade_boost_eligible = False
        payload.no_trade_boost_elapsed_minutes = 180

        decision = self.engine.evaluate(payload)
        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertAlmostEqual(float(decision.risk_pct), 0.02, places=6)
        self.assertFalse(bool(decision.diagnostics.get("risk_boost_active", False)))

    def test_soft_daily_dd_blocks_even_with_boost_when_not_elite(self) -> None:
        payload = self._base()
        payload.no_trade_boost_enabled = True
        payload.no_trade_boost_eligible = True
        payload.no_trade_boost_elapsed_minutes = 240
        payload.stats.daily_pnl_pct = -0.04

        decision = self.engine.evaluate(payload)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "daily_caution_quality_block")

    def test_live_budget_positive_when_effective_risk_positive(self) -> None:
        payload = self._base()
        payload.mode = "LIVE"
        payload.live_enabled = True
        payload.live_allowed = True
        payload.use_fixed_lot = True
        payload.fixed_lot = 0.01
        payload.requested_risk_pct = 0.02
        payload.hard_risk_cap = 0.02
        payload.stats.trades_today = 5
        payload.equity = 50.0
        payload.account_balance = 50.0
        payload.margin_free = 50.0
        decision = self.engine.evaluate(payload)
        self.assertEqual(decision.reason, "fixed_lot_risk_exceeds_budget")
        self.assertGreater(float(decision.risk_pct), 0.0)
        self.assertGreater(float(decision.diagnostics.get("budget_usd", 0.0)), 0.0)

    def test_xau_grid_min_lot_rounding_to_001_with_step(self) -> None:
        payload = self._base()
        payload.symbol = "XAUUSD"
        payload.setup = "XAUUSD_M5_GRID_SCALPER_START"
        payload.stats.trades_today = 5
        payload.entry_price = 2200.0
        payload.stop_price = 2196.0
        payload.equity = 50.0
        payload.margin_free = 100.0
        payload.requested_risk_pct = 0.10
        payload.hard_risk_cap = 0.005
        payload.strategy_risk_cap = 0.10
        payload.skip_micro_risk_clamp = True
        payload.micro_enabled = False
        payload.symbol_tick_size = 0.01
        payload.symbol_tick_value = 1.0
        payload.volume_min = 0.01
        payload.volume_step = 0.01
        payload.volume_max = 1.0
        decision = self.engine.evaluate(payload)
        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertAlmostEqual(float(decision.volume), 0.01, places=6)

    def test_kelly_zero_does_not_force_risk_zero(self) -> None:
        payload = self._base()
        payload.use_kelly = True
        payload.requested_risk_pct = 0.0025
        payload.hard_risk_cap = 0.005
        payload.stats.win_rate = 0.5  # zero Kelly edge
        payload.stats.avg_win_r = 1.0
        payload.stats.avg_loss_r = 1.0
        payload.stats.trades_today = 5
        decision = self.engine.evaluate(payload)
        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertGreater(float(decision.risk_pct), 0.0)

    def test_elite_trade_can_use_overflow_risk_band(self) -> None:
        payload = self._base()
        payload.use_fixed_lot = True
        payload.fixed_lot = 0.01
        payload.stats.trades_today = 5
        payload.current_phase = "PHASE_1"
        payload.current_base_risk_pct = 0.03
        payload.current_max_risk_pct = 0.05
        payload.requested_risk_pct = 0.03
        payload.hard_risk_cap = 0.05
        payload.session_multiplier = 1.15
        payload.ai_size_multiplier = 1.10
        payload.trade_quality_score = 0.93
        payload.trade_quality_band = "elite"
        payload.regime_confidence = 0.80
        payload.execution_quality_score = 0.92
        payload.execution_quality_state = "GOOD"
        payload.spread_quality_score = 0.90
        payload.session_quality_score = 0.95
        payload.recent_win_rate = 0.62
        payload.recent_expectancy_r = 0.18
        payload.news_state = "NEWS_SAFE"
        payload.news_confidence = 0.95

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertTrue(bool(decision.diagnostics.get("overflow_band_active", False)))
        self.assertGreater(float(decision.risk_pct), 0.03)

    def test_non_elite_trade_stays_within_base_band(self) -> None:
        payload = self._base()
        payload.use_fixed_lot = True
        payload.fixed_lot = 0.01
        payload.stats.trades_today = 5
        payload.current_phase = "PHASE_1"
        payload.current_base_risk_pct = 0.03
        payload.current_max_risk_pct = 0.05
        payload.requested_risk_pct = 0.03
        payload.hard_risk_cap = 0.05
        payload.session_multiplier = 1.20
        payload.ai_size_multiplier = 1.20
        payload.trade_quality_score = 0.74
        payload.trade_quality_band = "strong"
        payload.regime_confidence = 0.68
        payload.execution_quality_score = 0.82
        payload.execution_quality_state = "GOOD"
        payload.news_state = "NEWS_SAFE"
        payload.news_confidence = 0.90

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertFalse(bool(decision.diagnostics.get("overflow_band_active", False)))
        self.assertLessEqual(float(decision.risk_pct), 0.03 + 1e-9)

    def test_elite_trade_can_use_daily_trade_overflow_cap(self) -> None:
        payload = self._base()
        payload.use_fixed_lot = True
        payload.fixed_lot = 0.01
        payload.stats.trades_today = 4
        payload.max_trades_per_day = 4
        payload.overflow_max_trades_per_day = 7
        payload.current_phase = "PHASE_1"
        payload.current_base_risk_pct = 0.03
        payload.current_max_risk_pct = 0.05
        payload.requested_risk_pct = 0.03
        payload.hard_risk_cap = 0.05
        payload.trade_quality_score = 0.92
        payload.trade_quality_band = "elite"
        payload.regime_confidence = 0.80
        payload.execution_quality_score = 0.90
        payload.execution_quality_state = "GOOD"
        payload.spread_quality_score = 0.90
        payload.session_quality_score = 0.92
        payload.recent_win_rate = 0.61
        payload.recent_expectancy_r = 0.16
        payload.news_state = "NEWS_SAFE"
        payload.news_confidence = 0.95
        payload.stats.daily_pnl_pct = 0.02

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertLessEqual(float(decision.risk_pct), 0.03 + 1e-9)

    def test_non_elite_trade_can_use_stretch_capacity_after_daily_trade_cap(self) -> None:
        payload = self._base()
        payload.use_fixed_lot = True
        payload.fixed_lot = 0.01
        payload.stats.trades_today = 4
        payload.max_trades_per_day = 4
        payload.overflow_max_trades_per_day = 7
        payload.current_phase = "PHASE_1"
        payload.current_base_risk_pct = 0.03
        payload.current_max_risk_pct = 0.05
        payload.requested_risk_pct = 0.03
        payload.hard_risk_cap = 0.05
        payload.trade_quality_score = 0.79
        payload.regime_confidence = 0.70
        payload.execution_quality_score = 0.90
        payload.execution_quality_state = "GOOD"
        payload.spread_quality_score = 0.90
        payload.session_quality_score = 0.92
        payload.recent_win_rate = 0.61
        payload.recent_expectancy_r = 0.16
        payload.news_state = "NEWS_SAFE"
        payload.news_confidence = 0.95
        payload.stats.daily_pnl_pct = 0.02

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertFalse(bool(decision.diagnostics.get("overflow_band_active", False)))

    def test_soft_trade_budget_extends_daily_capacity_for_aligned_lane(self) -> None:
        payload = self._base()
        payload.use_fixed_lot = True
        payload.fixed_lot = 0.01
        payload.stats.trades_today = 6
        payload.max_trades_per_day = 4
        payload.overflow_max_trades_per_day = 4
        payload.stretch_max_trades_per_day = 6
        payload.hard_upper_limit = 8
        payload.soft_trade_budget_enabled = True
        payload.aggression_lane_multiplier = 1.18
        payload.execution_minute_quality_score = 0.82
        payload.execution_minute_size_multiplier = 1.12
        payload.microstructure_alignment_score = 0.38
        payload.microstructure_confidence = 0.74
        payload.lead_lag_alignment_score = 0.32
        payload.lead_lag_confidence = 0.70
        payload.event_playbook = "breakout"
        payload.event_pre_position_allowed = True
        payload.trade_quality_score = 0.78
        payload.regime_confidence = 0.74
        payload.execution_quality_score = 0.88
        payload.execution_quality_state = "GOOD"
        payload.spread_quality_score = 0.86
        payload.session_quality_score = 0.84
        payload.recent_win_rate = 0.58
        payload.recent_expectancy_r = 0.12
        payload.news_state = "NEWS_SAFE"
        payload.news_confidence = 0.90

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)

    def test_hot_lane_concurrency_bonus_raises_symbol_capacity(self) -> None:
        payload = self._base()
        payload.open_positions_symbol = 2
        payload.same_direction_positions = 2
        payload.max_positions_per_symbol = 2
        payload.hot_lane_concurrency_bonus = 1
        payload.hot_hand_active = True
        payload.microstructure_alignment_score = 0.30
        payload.trade_quality_score = 0.74
        payload.execution_quality_score = 0.82
        payload.execution_quality_state = "GOOD"
        payload.use_fixed_lot = True
        payload.fixed_lot = 0.01
        payload.entry_price = 2200.0
        payload.stop_price = 2199.7

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)

    def test_qualified_b_setup_is_part_of_normal_runtime_flow(self) -> None:
        payload = self._base()
        payload.trade_quality_score = 0.60
        payload.trade_quality_band = "acceptable"
        payload.trade_quality_detail = "B"
        payload.quality_size_multiplier = 0.65
        payload.ai_probability = 0.62
        payload.expected_value_r = 0.24
        payload.confluence_score = 3.2
        payload.execution_quality_score = 0.82
        payload.execution_quality_state = "GOOD"
        payload.spread_quality_score = 0.80
        payload.session_quality_score = 0.78
        payload.news_state = "NEWS_SAFE"
        payload.news_confidence = 0.90

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertEqual(str(decision.diagnostics.get("daily_state")), "DAILY_NORMAL")
        self.assertLess(float(decision.risk_pct), 0.005)

    def test_daily_defensive_still_allows_strong_b_plus_flow(self) -> None:
        payload = self._base()
        payload.stats.daily_pnl_pct = -0.055
        payload.trade_quality_score = 0.74
        payload.trade_quality_band = "strong"
        payload.trade_quality_detail = "B+"
        payload.quality_size_multiplier = 0.95
        payload.ai_probability = 0.74
        payload.expected_value_r = 0.35
        payload.confluence_score = 3.8
        payload.execution_quality_score = 0.86
        payload.execution_quality_state = "GOOD"
        payload.spread_quality_score = 0.82
        payload.session_quality_score = 0.80
        payload.news_state = "NEWS_SAFE"
        payload.news_confidence = 0.92

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertEqual(str(decision.diagnostics.get("daily_state")), "DAILY_DEFENSIVE")
        self.assertEqual(str(decision.diagnostics.get("current_capacity_mode")), "DEFENSIVE_FLOW")

    def test_session_priority_modestly_increases_lane_capacity_without_breaking_caps(self) -> None:
        payload = self._base()
        payload.max_trades_per_day = 10
        payload.overflow_max_trades_per_day = 16
        payload.stretch_max_trades_per_day = 16
        payload.hard_upper_limit = 20
        payload.cluster_mode_active = True
        payload.trade_quality_score = 0.78
        payload.trade_quality_band = "strong"
        payload.trade_quality_detail = "B+"
        payload.quality_size_multiplier = 0.95
        payload.execution_quality_score = 0.88
        payload.execution_quality_state = "GOOD"
        payload.news_state = "NEWS_SAFE"
        payload.news_confidence = 0.92
        payload.session_priority_multiplier = 1.12
        payload.lane_budget_share = 0.40

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertGreaterEqual(int(decision.diagnostics.get("lane_available_capacity", 0)), 5)
        self.assertLessEqual(int(decision.diagnostics.get("projected_trade_capacity_today", 0)), 20)

    def test_hot_hand_profit_recycle_and_session_allocator_press_proven_lane(self) -> None:
        neutral = self._base()
        neutral.max_trades_per_day = 12
        neutral.overflow_max_trades_per_day = 18
        neutral.stretch_max_trades_per_day = 18
        neutral.hard_upper_limit = 22
        neutral.max_trades_per_hour = 4
        neutral.stretch_max_trades_per_hour = 6
        neutral.cluster_mode_active = True
        neutral.trade_quality_score = 0.80
        neutral.trade_quality_band = "strong"
        neutral.trade_quality_detail = "A-"
        neutral.quality_size_multiplier = 0.98
        neutral.execution_quality_score = 0.90
        neutral.execution_quality_state = "GOOD"
        neutral.spread_quality_score = 0.84
        neutral.session_quality_score = 0.84
        neutral.news_state = "NEWS_SAFE"
        neutral.news_confidence = 0.94
        neutral.session_priority_multiplier = 1.02
        neutral.lane_budget_share = 0.22
        neutral.lane_expectancy_multiplier = 1.08
        neutral.lane_expectancy_score = 0.62
        neutral.trade_management_state = "ACTIVE"

        boosted = self._base()
        boosted.max_trades_per_day = 12
        boosted.overflow_max_trades_per_day = 18
        boosted.stretch_max_trades_per_day = 18
        boosted.hard_upper_limit = 22
        boosted.max_trades_per_hour = 4
        boosted.stretch_max_trades_per_hour = 6
        boosted.cluster_mode_active = True
        boosted.trade_quality_score = 0.80
        boosted.trade_quality_band = "strong"
        boosted.trade_quality_detail = "A-"
        boosted.quality_size_multiplier = 0.98
        boosted.execution_quality_score = 0.90
        boosted.execution_quality_state = "GOOD"
        boosted.spread_quality_score = 0.84
        boosted.session_quality_score = 0.84
        boosted.news_state = "NEWS_SAFE"
        boosted.news_confidence = 0.94
        boosted.session_priority_multiplier = 1.14
        boosted.lane_budget_share = 0.28
        boosted.lane_expectancy_multiplier = 1.18
        boosted.lane_expectancy_score = 0.74
        boosted.hot_hand_active = True
        boosted.hot_hand_score = 0.78
        boosted.session_bankroll_bias = 1.20
        boosted.profit_recycle_active = True
        boosted.profit_recycle_boost = 0.10
        boosted.close_winners_score = 0.76
        boosted.trade_management_state = "ACTIVE"

        neutral_decision = self.engine.evaluate(neutral)
        boosted_decision = self.engine.evaluate(boosted)

        self.assertTrue(neutral_decision.approved, msg=neutral_decision.reason)
        self.assertTrue(boosted_decision.approved, msg=boosted_decision.reason)
        self.assertGreater(
            int(boosted_decision.diagnostics.get("projected_trade_capacity_today") or 0),
            int(neutral_decision.diagnostics.get("projected_trade_capacity_today") or 0),
        )
        self.assertGreater(
            int(boosted_decision.diagnostics.get("lane_available_capacity") or 0),
            int(neutral_decision.diagnostics.get("lane_available_capacity") or 0),
        )
        self.assertGreater(
            float(boosted_decision.diagnostics.get("session_priority_multiplier") or 0.0),
            float(neutral_decision.diagnostics.get("session_priority_multiplier") or 0.0),
        )
        self.assertTrue(bool(boosted_decision.diagnostics.get("hot_hand_active")))
        self.assertTrue(bool(boosted_decision.diagnostics.get("profit_recycle_active")))

    def test_hot_lane_borrow_share_expands_capacity_for_proven_fast_lane(self) -> None:
        baseline = self._base()
        baseline.max_trades_per_day = 12
        baseline.stretch_max_trades_per_day = 16
        baseline.hard_upper_limit = 20
        baseline.max_trades_per_hour = 4
        baseline.stretch_max_trades_per_hour = 6
        baseline.trade_quality_score = 0.78
        baseline.execution_quality_score = 0.88
        baseline.execution_quality_state = "GOOD"
        baseline.session_priority_multiplier = 1.08
        baseline.lane_budget_share = 0.24
        baseline.lane_expectancy_multiplier = 1.10
        baseline.lane_expectancy_score = 0.62

        borrowed = self._base()
        borrowed.max_trades_per_day = 12
        borrowed.stretch_max_trades_per_day = 16
        borrowed.hard_upper_limit = 20
        borrowed.max_trades_per_hour = 4
        borrowed.stretch_max_trades_per_hour = 6
        borrowed.trade_quality_score = 0.82
        borrowed.execution_quality_score = 0.90
        borrowed.execution_quality_state = "GOOD"
        borrowed.session_priority_multiplier = 1.16
        borrowed.lane_budget_share = 0.28
        borrowed.lane_expectancy_multiplier = 1.18
        borrowed.lane_expectancy_score = 0.74
        borrowed.hot_hand_active = True
        borrowed.hot_hand_score = 0.80
        borrowed.hot_lane_concurrency_bonus = 2

        baseline_decision = self.engine.evaluate(baseline)
        borrowed_decision = self.engine.evaluate(borrowed)

        self.assertTrue(borrowed_decision.approved, msg=borrowed_decision.reason)
        self.assertGreater(float(borrowed_decision.diagnostics.get("hot_lane_borrow_share") or 0.0), 0.0)
        self.assertGreater(
            int(borrowed_decision.diagnostics.get("lane_available_capacity") or 0),
            int(baseline_decision.diagnostics.get("lane_available_capacity") or 0),
        )

    def test_winning_lane_can_extend_hourly_capacity_beyond_stretch_target(self) -> None:
        payload = self._base()
        payload.max_trades_per_day = 20
        payload.overflow_max_trades_per_day = 32
        payload.stretch_max_trades_per_day = 28
        payload.hard_upper_limit = 36
        payload.max_trades_per_hour = 4
        payload.stretch_max_trades_per_hour = 6
        payload.trade_quality_score = 0.76
        payload.execution_quality_score = 0.88
        payload.execution_quality_state = "GOOD"
        payload.spread_quality_score = 0.84
        payload.session_quality_score = 0.82
        payload.news_state = "NEWS_SAFE"
        payload.news_confidence = 0.90
        payload.recent_expectancy_r = 0.08
        payload.winning_streak_mode_active = True
        payload.stats.winning_streak = 3
        payload.hot_lane_concurrency_bonus = 2

        decision = self.engine.evaluate(payload)

        self.assertTrue(decision.approved, msg=decision.reason)
        self.assertGreaterEqual(int(decision.diagnostics.get("effective_hourly_trade_cap") or 0), 8)


if __name__ == "__main__":
    unittest.main()
