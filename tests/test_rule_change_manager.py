from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.rule_change_manager import RuleChangeConfig, RuleChangeManager

UTC = timezone.utc


class RuleChangeManagerTests(unittest.TestCase):
    def test_does_not_apply_without_sample_or_shadow(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            manager = RuleChangeManager(
                db_path=Path(tmp_dir) / "rule_changes.sqlite",
                config=RuleChangeConfig(enabled=True, min_samples=5, shadow_trades_required=3, cooldown_hours=24),
            )
            decision = manager.evaluate_and_apply(
                strategy="XAUUSD_M5_GRID",
                baseline_metrics={"profit_factor": 1.1, "expectancy_r": 0.05, "win_rate": 0.5, "max_drawdown_pct": 0.08},
                candidate_metrics={"profit_factor": 1.3, "expectancy_r": 0.1, "win_rate": 0.6, "max_drawdown_pct": 0.07},
                current_params={"grid_step": 35.0},
                proposed_params={"grid_step": 33.0},
                shadow_trades=0,
                now_utc=datetime.now(tz=UTC),
            )
            self.assertFalse(decision.approved)
            self.assertEqual(decision.reason, "insufficient_samples")

    def test_applies_when_improvement_bounded_and_cooldown_passed(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            manager = RuleChangeManager(
                db_path=Path(tmp_dir) / "rule_changes.sqlite",
                config=RuleChangeConfig(
                    enabled=True,
                    min_samples=5,
                    shadow_trades_required=3,
                    cooldown_hours=1,
                    max_delta_pct=0.2,
                    allowed_params=("grid_step", "risk_pct"),
                ),
            )
            for _ in range(6):
                manager.record_trade("XAUUSD_M5_GRID", pnl_r=0.2, mode="LIVE")
            decision = manager.evaluate_and_apply(
                strategy="XAUUSD_M5_GRID",
                baseline_metrics={"profit_factor": 1.2, "expectancy_r": 0.05, "win_rate": 0.52, "max_drawdown_pct": 0.09},
                candidate_metrics={"profit_factor": 1.3, "expectancy_r": 0.08, "win_rate": 0.56, "max_drawdown_pct": 0.08},
                current_params={"grid_step": 35.0, "risk_pct": 0.08},
                proposed_params={"grid_step": 32.0, "risk_pct": 0.09},
                shadow_trades=5,
                now_utc=datetime.now(tz=UTC),
            )
            self.assertTrue(decision.approved)
            self.assertEqual(decision.reason, "applied")
            self.assertEqual(decision.version, 1)
            latest = manager.latest_decision("XAUUSD_M5_GRID")
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(int(latest["version"]), 1)

            cooldown_decision = manager.evaluate_and_apply(
                strategy="XAUUSD_M5_GRID",
                baseline_metrics={"profit_factor": 1.2, "expectancy_r": 0.05, "win_rate": 0.52, "max_drawdown_pct": 0.09},
                candidate_metrics={"profit_factor": 1.35, "expectancy_r": 0.10, "win_rate": 0.58, "max_drawdown_pct": 0.08},
                current_params={"grid_step": 32.0},
                proposed_params={"grid_step": 31.0},
                shadow_trades=6,
                now_utc=datetime.now(tz=UTC) + timedelta(minutes=10),
            )
            self.assertFalse(cooldown_decision.approved)
            self.assertEqual(cooldown_decision.reason, "cooldown_active")

    def test_locked_params_cannot_be_changed(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            manager = RuleChangeManager(
                db_path=Path(tmp_dir) / "rule_changes.sqlite",
                config=RuleChangeConfig(
                    enabled=True,
                    min_samples=3,
                    shadow_trades_required=2,
                    cooldown_hours=1,
                    allowed_params=("min_confluence_score", "xau_grid_max_legs"),
                    locked_params=("xau_grid_max_legs",),
                ),
            )
            for _ in range(4):
                manager.record_trade("XAUUSD_M5_GRID", pnl_r=0.3, mode="LIVE")
            decision = manager.evaluate_and_apply(
                strategy="XAUUSD_M5_GRID",
                baseline_metrics={"profit_factor": 1.0, "expectancy_r": 0.02, "win_rate": 0.5, "max_drawdown_pct": 0.10},
                candidate_metrics={"profit_factor": 1.2, "expectancy_r": 0.05, "win_rate": 0.55, "max_drawdown_pct": 0.09},
                current_params={"xau_grid_max_legs": 8},
                proposed_params={"xau_grid_max_legs": 9},
                shadow_trades=3,
                now_utc=datetime.now(tz=UTC),
            )
            self.assertFalse(decision.approved)
            self.assertEqual(decision.reason, "param_locked:xau_grid_max_legs")


if __name__ == "__main__":
    unittest.main()
