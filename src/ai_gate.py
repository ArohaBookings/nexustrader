from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import json

import joblib
import pandas as pd

from src.openai_client import OpenAIClient
from src.strategy_engine import SignalCandidate
from src.trade_management import build_local_management_plan, build_local_trade_plan
from src.utils import clamp


@dataclass
class AIGateDecision:
    approved: bool
    probability: float
    expected_value_r: float
    size_multiplier: float
    sl_multiplier: float
    tp_r: float
    reason: str


@dataclass
class AIGate:
    scorer_path: Path
    value_path: Path
    risk_modulator_path: Path
    schema_path: Path
    min_probabilities: dict[str, float]
    enabled: bool = True
    remote_enabled: bool = True
    live_remote_enabled: bool = False
    openai_api_env: str = "OPENAI_API_KEY"
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 8.0
    openai_retry_once: bool = False
    remote_score_enabled: bool = False
    logger: Any | None = None

    def __post_init__(self) -> None:
        self._utc = timezone.utc
        self._grid_approval_cache_ttl_seconds = 20
        self._grid_approval_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}
        self._scorer = self._safe_load(self.scorer_path)
        self._value_model = self._safe_load(self.value_path)
        self._risk_model = self._safe_load(self.risk_modulator_path)
        self._schema = self._load_schema()
        self._openai = OpenAIClient(
            api_key_env=self.openai_api_env,
            model=self.openai_model,
            timeout_seconds=self.openai_timeout_seconds,
            retry_once=self.openai_retry_once,
            enabled=False,
            logger=self.logger,
        )

    def health(self) -> dict[str, Any]:
        return {
            "ok": bool(self.enabled),
            "mode": "local",
            "last_error": "live_remote_disabled",
            "model": self.openai_model,
        }

    def ai_test(self) -> tuple[bool, str]:
        return False, "live_remote_disabled"

    def approve_grid_cycle(self, state: dict[str, Any]) -> dict[str, Any]:
        now_ts = datetime.now(tz=self._utc)
        cache_key = self._grid_cache_key(state)
        cached = self._grid_approval_cache.get(cache_key)
        if cached is not None:
            cached_at, cached_payload = cached
            if (now_ts - cached_at).total_seconds() <= float(self._grid_approval_cache_ttl_seconds):
                output = dict(cached_payload)
                output["cache_hit"] = True
                return output

        ai_mode = "local_brain"

        confluence = clamp(float(state.get("confluence", 0.0)), 0.0, 5.0)
        spread_points = max(0.0, float(state.get("spread_points", 0.0)))
        atr = max(1e-6, float(state.get("atr", 0.0)))
        ema_dev_atr = abs(float(state.get("ema_dev_atr", 0.0)))
        news_safe = bool(state.get("news_safe", True))
        probability = 0.46 + (0.07 * confluence) + min(0.1, ema_dev_atr * 0.05)
        if spread_points > 0:
            probability -= min(0.10, (spread_points / max(atr * 100.0, 1.0)) * 0.03)
        if not news_safe:
            probability -= 0.04
        probability = clamp(probability, 0.0, 1.0)
        threshold = 0.58 if news_safe else 0.72
        approve = probability >= threshold
        reason = f"local_grid_p={probability:.2f}"
        output = {
            "approve": approve,
            "confidence": probability,
            "reason": reason,
            "lot_multiplier": 1.0 if approve else 0.5,
            "step_multiplier": 1.0 if approve else 1.2,
            "max_levels": state.get("max_levels"),
            "ai_mode": ai_mode,
        }
        self._grid_approval_cache[cache_key] = (now_ts, dict(output))
        return output

    @staticmethod
    def _grid_cache_key(state: dict[str, Any]) -> str:
        symbol = str(state.get("symbol", "XAUUSD")).upper()
        mode = str(state.get("mode", "NEW_CYCLE")).upper()
        side = str(state.get("side", "")).upper()
        spread_bucket = int(clamp(float(state.get("spread_points", 0.0)), 0.0, 500.0))
        atr_bucket = int(clamp(float(state.get("atr", 0.0)) * 1000.0, 0.0, 1_000_000.0))
        stretch_bucket = int(clamp(abs(float(state.get("ema_dev_atr", 0.0))) * 100.0, 0.0, 1000.0))
        return f"{symbol}:{mode}:{side}:{spread_bucket}:{atr_bucket}:{stretch_bucket}"

    def evaluate(
        self,
        candidate: SignalCandidate,
        row: pd.Series,
        regime: str,
        consecutive_losses: int,
    ) -> AIGateDecision:
        if not self.enabled:
            return AIGateDecision(True, 1.0, candidate.tp_r, 1.0, 1.0, candidate.tp_r, "AI disabled")

        probability, expected_value, score_source = self._score_trade(candidate, row, regime)
        size_multiplier, sl_multiplier, tp_r = self._modulate_risk(candidate, row, regime, consecutive_losses, probability)
        required_probability = self.min_probabilities.get(candidate.setup, self.min_probabilities.get("DEFAULT", 0.58))
        quality_tier = str((candidate.meta or {}).get("quality_tier", "")).upper()
        throughput_recovery = bool((candidate.meta or {}).get("throughput_recovery_active", False))
        frequency_catchup_pressure = clamp(float((candidate.meta or {}).get("frequency_catchup_pressure", 0.0) or 0.0), 0.0, 1.0)
        soft_burst_target_10m = max(0, int((candidate.meta or {}).get("soft_burst_target_10m", 0) or 0))
        aggressive_reentry_enabled = bool((candidate.meta or {}).get("aggressive_reentry_enabled", False))
        undertrade_fix_mode = bool((candidate.meta or {}).get("undertrade_fix_mode", False))
        meta = dict(candidate.meta or {})
        symbol_key = "".join(
            char
            for char in str(
                meta.get("symbol_key")
                or meta.get("symbol")
                or row.get("symbol_key")
                or row.get("symbol")
                or ""
            ).upper()
            if char.isalnum()
        )
        session_name = str(
            meta.get("session_name")
            or meta.get("session")
            or row.get("session_name")
            or row.get("session")
            or ""
        ).upper()
        session_native_pair = bool(meta.get("session_native_pair", False))
        gpt_hybrid_active = bool(meta.get("gpt_hybrid_active", False) and meta.get("gpt_hybrid_session_match", False))
        gpt_hybrid_conviction = clamp(float(meta.get("gpt_hybrid_conviction", 0.0) or 0.0), 0.0, 1.0)
        gpt_hybrid_threshold_delta = clamp(float(meta.get("gpt_hybrid_threshold_delta", 0.0) or 0.0), -0.08, 0.05)
        normalized_confluence = clamp(float(candidate.confluence_score or 0.0) / 5.0, 0.0, 1.0)
        if candidate.setup in {"BREAKOUT_RETEST", "TREND_CONTINUATION"} and normalized_confluence < 0.76:
            required_probability += 0.02
        if quality_tier == "B" and not throughput_recovery:
            required_probability += 0.015
        if quality_tier in {"A", "A+"} and throughput_recovery:
            required_probability -= 0.01
        if frequency_catchup_pressure >= 0.40 and quality_tier in {"A", "A+"}:
            required_probability -= min(0.02, frequency_catchup_pressure * 0.03)
        if (
            undertrade_fix_mode
            and symbol_key == "BTCUSD"
            and soft_burst_target_10m >= 6
            and normalized_confluence >= 0.72
        ):
            required_probability -= 0.015
        if (
            undertrade_fix_mode
            and symbol_key == "XAUUSD"
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and soft_burst_target_10m >= 6
            and normalized_confluence >= 0.72
        ):
            required_probability -= 0.015
        asia_native_pair = symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD", "USDJPY", "EURJPY", "GBPJPY"}
        if (
            undertrade_fix_mode
            and asia_native_pair
            and session_native_pair
            and session_name in {"SYDNEY", "TOKYO"}
            and soft_burst_target_10m >= 5
            and normalized_confluence >= 0.68
        ):
            required_probability -= 0.02
        if aggressive_reentry_enabled and normalized_confluence >= 0.78:
            required_probability -= 0.01
        if gpt_hybrid_active and gpt_hybrid_conviction >= 0.45 and normalized_confluence >= 0.70:
            required_probability += gpt_hybrid_threshold_delta
        required_probability = clamp(required_probability, 0.35, 0.90)
        approved = probability >= required_probability and expected_value >= self.min_probabilities.get("EV", 0.10)
        reason = f"p={probability:.2f} ev={expected_value:.2f} regime={regime} source={score_source}"
        if gpt_hybrid_active and gpt_hybrid_conviction >= 0.45:
            reason = f"{reason} hybrid={gpt_hybrid_conviction:.2f}"
        return AIGateDecision(approved, probability, expected_value, size_multiplier, sl_multiplier, tp_r, reason)

    def approve_order(
        self,
        candidate: SignalCandidate,
        features: pd.Series,
        regime: str,
        risk_summary: dict[str, Any],
        news_summary: dict[str, Any],
        open_positions: list[dict[str, Any]],
        account_state: dict[str, Any],
        precomputed_gate: AIGateDecision | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        risk_ok = bool(risk_summary.get("approved", False))
        news_safe = bool(news_summary.get("safe", False))
        news_state = str(
            news_summary.get(
                "state",
                "NEWS_SAFE" if news_safe else "NEWS_BLOCKED",
            )
        ).upper()
        default_payload = {
            "approve": False,
            "confidence": 0.0,
            "probability": 0.0,
            "recommended_sl_atr_mult": candidate.stop_atr,
            "recommended_tp_r": candidate.tp_r,
            "trailing_enabled": False,
            "partial_close_enabled": False,
            "max_slippage_points": None,
            "size_multiplier": 0.0,
            "rationale": "blocked_by_upstream_gate",
            "ai_mode": "local_fallback",
        }
        if not risk_ok:
            default_payload["rationale"] = f"risk_not_ok:{risk_summary.get('reason', 'unknown')}"
            return False, default_payload
        if not news_safe or news_state == "NEWS_BLOCKED":
            default_payload["rationale"] = f"news_not_safe:{news_summary.get('reason', 'unknown')}"
            return False, default_payload

        tier_decision = precomputed_gate or self.evaluate(
            candidate=candidate,
            row=features,
            regime=regime,
            consecutive_losses=int(risk_summary.get("consecutive_losses", 0)),
        )
        caution_mode = news_state == "NEWS_CAUTION"
        caution_probability_floor = clamp(
            float(news_summary.get("caution_probability_floor", 0.0) or 0.0),
            0.0,
            1.0,
        )
        caution_confluence_floor = clamp(
            float(news_summary.get("caution_confluence_floor", 0.0) or 0.0),
            0.0,
            1.0,
        )
        if caution_mode:
            normalized_confluence = clamp(float(candidate.confluence_score) / 5.0, 0.0, 1.0)
            if tier_decision.probability < caution_probability_floor or normalized_confluence < caution_confluence_floor:
                default_payload["rationale"] = f"news_caution_gate:{news_summary.get('reason', 'unknown')}"
                return False, default_payload

        if self._risk_model is not None and self.enabled:
            try:
                vector = self._vectorize(features)
                raw = self._risk_model.predict(vector)[0]
                if isinstance(raw, (list, tuple)) and len(raw) >= 3:
                    sl_atr_mult = clamp(float(raw[1]) * candidate.stop_atr, 0.8, 2.0)
                    tp_r = clamp(float(raw[2]), 1.5, 2.5)
                    size_multiplier = clamp(float(raw[0]), 0.25, 1.0)
                else:
                    sl_atr_mult = clamp(candidate.stop_atr * tier_decision.sl_multiplier, 0.8, 2.0)
                    tp_r = clamp(tier_decision.tp_r, 1.5, 2.5)
                    size_multiplier = clamp(float(raw), 0.25, 1.0)
            except Exception:
                sl_atr_mult, tp_r, size_multiplier = self._fallback_final_order(candidate, features, regime, tier_decision, open_positions, risk_summary)
        else:
            sl_atr_mult, tp_r, size_multiplier = self._fallback_final_order(candidate, features, regime, tier_decision, open_positions, risk_summary)

        spread_points = float(risk_summary.get("spread_points", 0.0))
        max_spread_points = float(risk_summary.get("max_spread_points", spread_points or 999.0))
        exposure_cap = float(risk_summary.get("portfolio_size_multiplier", 1.0))
        size_multiplier = clamp(size_multiplier * exposure_cap, 0.0, 1.0)
        if caution_mode:
            size_multiplier = clamp(size_multiplier * 0.85, 0.0, 1.0)
        adaptive = self._adaptive_overrides(candidate, risk_summary)
        size_multiplier = clamp(size_multiplier * adaptive["size_multiplier"], 0.0, 1.0)
        session_probability_offset = float(risk_summary.get("session_probability_offset", 0.0))
        base_probability_floor = self.min_probabilities.get(candidate.setup, self.min_probabilities.get("DEFAULT", 0.58))
        adaptive_probability_floor = max(
            0.0,
            base_probability_floor + adaptive["probability_delta"] + session_probability_offset,
        )
        effective_probability = tier_decision.probability
        remote_fail_open = False
        ai_mode = "local_brain"
        remote_advice: dict[str, Any] = {}
        trail_mult = None

        bias_direction = str(news_summary.get("bias_direction", "neutral")).lower()
        bias_confidence = clamp(float(news_summary.get("bias_confidence", 0.0)), 0.0, 1.0)
        source_confidence = clamp(float(news_summary.get("source_confidence", 0.5)), 0.0, 1.0)
        authenticity_risk = clamp(float(news_summary.get("authenticity_risk", 0.0)), 0.0, 1.0)
        sentiment_extreme = clamp(float(news_summary.get("sentiment_extreme", 0.0)), 0.0, 1.0)
        crowding_bias = str(news_summary.get("crowding_bias", "neutral")).lower()
        dxy_support_score = clamp(
            float(
                risk_summary.get(
                    "dxy_support_score",
                    features.get("dxy_support_score", 0.5) if hasattr(features, "get") else 0.5,
                )
                or 0.5
            ),
            0.0,
            1.0,
        )
        meta = dict(candidate.meta or {})
        symbol_key = "".join(
            char
            for char in str(
                meta.get("symbol_key")
                or meta.get("symbol")
                or features.get("symbol_key")
                or features.get("symbol")
                or ""
            ).upper()
            if char.isalnum()
        )
        session_name = str(
            meta.get("session_name")
            or meta.get("session")
            or features.get("session_name")
            or features.get("session")
            or ""
        ).upper()
        session_native_pair = bool(meta.get("session_native_pair", False))
        undertrade_fix_mode = bool(meta.get("undertrade_fix_mode", False))
        soft_burst_target_10m = max(0, int(meta.get("soft_burst_target_10m", 0) or 0))
        asia_native_pair = symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD", "USDJPY", "EURJPY", "GBPJPY"}
        trend_like_setup = any(token in str(candidate.setup).upper() for token in ("TREND", "BREAKOUT", "CONTINUATION", "IMPULSE"))
        if bias_direction in {"bullish", "bearish"} and bias_confidence > 0:
            aligned = (bias_direction == "bullish" and candidate.side == "BUY") or (bias_direction == "bearish" and candidate.side == "SELL")
            if aligned:
                effective_probability = clamp(effective_probability + (0.015 * bias_confidence), 0.0, 1.0)
            else:
                effective_probability = clamp(effective_probability - (0.03 * bias_confidence), 0.0, 1.0)
            if aligned and bias_confidence >= 0.70 and source_confidence >= 0.60 and authenticity_risk <= 0.35:
                effective_probability = clamp(effective_probability + 0.015, 0.0, 1.0)
                size_multiplier = clamp(size_multiplier * 1.03, 0.0, 1.0)
            elif (not aligned) and bias_confidence >= 0.70 and source_confidence >= 0.60 and authenticity_risk <= 0.35:
                effective_probability = clamp(effective_probability - 0.03, 0.0, 1.0)
        if authenticity_risk > 0:
            effective_probability = clamp(effective_probability - (0.04 * authenticity_risk), 0.0, 1.0)
        if source_confidence < 0.5:
            effective_probability = clamp(effective_probability - ((0.5 - source_confidence) * 0.04), 0.0, 1.0)
        if sentiment_extreme >= 0.75 and crowding_bias in {"bullish", "bearish"}:
            is_crowded_side = (crowding_bias == "bullish" and candidate.side == "BUY") or (crowding_bias == "bearish" and candidate.side == "SELL")
            if is_crowded_side:
                effective_probability = clamp(effective_probability - 0.02, 0.0, 1.0)
            else:
                effective_probability = clamp(effective_probability + 0.01, 0.0, 1.0)
        if trend_like_setup:
            if dxy_support_score < 0.45:
                effective_probability = clamp(effective_probability - ((0.45 - dxy_support_score) * 0.08), 0.0, 1.0)
            elif dxy_support_score >= 0.65:
                effective_probability = clamp(effective_probability + min(0.02, (dxy_support_score - 0.65) * 0.08), 0.0, 1.0)

        slippage_quality_score = clamp(
            float(
                risk_summary.get(
                    "slippage_quality_score",
                    risk_summary.get("execution_quality_score", 0.55),
                )
                or 0.55
            ),
            0.0,
            1.0,
        )
        execution_quality_state = str(
            risk_summary.get(
                "execution_quality_state",
                risk_summary.get("execution_quality_state_runtime", "GOOD"),
            )
            or "GOOD"
        ).upper()
        execution_spread_ema_points = max(
            0.0,
            float(
                risk_summary.get(
                    "execution_spread_ema_points",
                    risk_summary.get("spread_ema_points", risk_summary.get("spread_points", 0.0)),
                )
                or 0.0
            ),
        )
        hour_expectancy_score = clamp(
            float(
                risk_summary.get(
                    "hour_expectancy_score",
                    risk_summary.get("lane_expectancy_score", 0.5),
                )
                or 0.5
            ),
            0.0,
            1.0,
        )
        lane_expectancy_multiplier = clamp(
            float(risk_summary.get("lane_expectancy_multiplier", 1.0) or 1.0),
            0.75,
            1.40,
        )
        fill_pressure = execution_spread_ema_points / max(max_spread_points, 1.0)
        microstructure_ready = bool(risk_summary.get("microstructure_ready", False))
        microstructure_direction = str(risk_summary.get("microstructure_direction", "neutral") or "neutral").lower()
        microstructure_confidence = clamp(float(risk_summary.get("microstructure_confidence", 0.0) or 0.0), 0.0, 1.0)
        microstructure_pressure_score = float(risk_summary.get("microstructure_pressure_score", 0.0) or 0.0)
        microstructure_depth_imbalance = float(risk_summary.get("microstructure_depth_imbalance", 0.0) or 0.0)
        microstructure_drift_score = float(risk_summary.get("microstructure_drift_score", 0.0) or 0.0)
        microstructure_spread_stability = clamp(float(risk_summary.get("microstructure_spread_stability", 0.0) or 0.0), 0.0, 1.0)
        probability_floor_adjustment = 0.0
        if (
            undertrade_fix_mode
            and asia_native_pair
            and session_native_pair
            and session_name in {"SYDNEY", "TOKYO"}
            and soft_burst_target_10m >= 5
        ):
            probability_floor_adjustment -= 0.015
            if slippage_quality_score >= 0.55 and execution_quality_state == "GOOD":
                size_multiplier = clamp(size_multiplier * 1.04, 0.0, 1.0)
        if microstructure_ready and microstructure_confidence >= 0.55:
            aligned = (
                (microstructure_direction == "bullish" and candidate.side == "BUY")
                or (microstructure_direction == "bearish" and candidate.side == "SELL")
            )
            micro_signal_strength = clamp(
                (abs(microstructure_pressure_score) * 0.55)
                + (abs(microstructure_depth_imbalance) * 0.20)
                + (abs(microstructure_drift_score) * 0.15)
                + (microstructure_spread_stability * 0.10),
                0.0,
                1.0,
            )
            if aligned:
                effective_probability = clamp(
                    effective_probability + (0.010 + (microstructure_confidence * 0.02) + (micro_signal_strength * 0.01)),
                    0.0,
                    1.0,
                )
                if fill_pressure <= 0.90 and slippage_quality_score >= 0.55:
                    size_multiplier = clamp(size_multiplier * min(1.08, 1.02 + (micro_signal_strength * 0.05)), 0.0, 1.0)
                if symbol_key == "XAUUSD" and "GRID" in str(candidate.setup).upper():
                    tp_r = clamp(tp_r + 0.08 + (micro_signal_strength * 0.04), 1.40, 3.20)
            elif microstructure_direction in {"bullish", "bearish"}:
                effective_probability = clamp(
                    effective_probability - (0.015 + (microstructure_confidence * 0.02) + (micro_signal_strength * 0.02)),
                    0.0,
                    1.0,
                )
                probability_floor_adjustment += 0.02
                size_multiplier = clamp(size_multiplier * 0.86, 0.0, 1.0)
            if microstructure_spread_stability <= 0.40:
                probability_floor_adjustment += 0.01
                size_multiplier = clamp(size_multiplier * 0.94, 0.0, 1.0)

        adaptive_guard = True
        adaptive_samples = float(risk_summary.get("adaptive_samples", 0.0))
        adaptive_win_rate = float(risk_summary.get("adaptive_win_rate", 0.5))
        adaptive_loss_streak = int(float(risk_summary.get("adaptive_recent_loss_streak", 0.0)))
        if adaptive_samples >= 8 and adaptive_win_rate < 0.45 and adaptive_loss_streak >= 2:
            adaptive_guard = effective_probability >= 0.86

        adaptive_min_floor = clamp(float(risk_summary.get("adaptive_min_probability_floor", 0.50)), 0.35, 0.90)
        required_probability = max(adaptive_min_floor, min(0.9, adaptive_probability_floor))
        required_probability = clamp(required_probability + probability_floor_adjustment, 0.35, 0.90)
        if trend_like_setup and dxy_support_score < 0.45:
            required_probability = clamp(required_probability + 0.02, 0.35, 0.90)
        if (
            bias_direction in {"bullish", "bearish"}
            and bias_confidence >= 0.70
            and source_confidence >= 0.60
            and authenticity_risk <= 0.35
        ):
            aligned = (bias_direction == "bullish" and candidate.side == "BUY") or (bias_direction == "bearish" and candidate.side == "SELL")
            if aligned:
                required_probability = clamp(required_probability - 0.01, 0.35, 0.90)
            else:
                required_probability = clamp(required_probability + 0.03, 0.35, 0.90)
        if slippage_quality_score <= 0.40:
            effective_probability = clamp(effective_probability - 0.03, 0.0, 1.0)
            required_probability = clamp(required_probability + 0.02, 0.35, 0.90)
            size_multiplier = clamp(size_multiplier * 0.88, 0.0, 1.0)
        elif slippage_quality_score >= 0.72 and fill_pressure <= 0.75:
            effective_probability = clamp(effective_probability + 0.01, 0.0, 1.0)
            if lane_expectancy_multiplier >= 1.02:
                size_multiplier = clamp(size_multiplier * 1.03, 0.0, 1.0)
        if execution_quality_state in {"DEGRADED", "POOR"}:
            effective_probability = clamp(effective_probability - 0.025, 0.0, 1.0)
            required_probability = clamp(required_probability + 0.015, 0.35, 0.90)
            size_multiplier = clamp(size_multiplier * 0.84, 0.0, 1.0)
        if hour_expectancy_score >= 0.62:
            effective_probability = clamp(effective_probability + 0.01, 0.0, 1.0)
            if slippage_quality_score >= 0.55 and execution_quality_state == "GOOD":
                size_multiplier = clamp(size_multiplier * min(1.05, lane_expectancy_multiplier), 0.0, 1.0)
        elif hour_expectancy_score <= 0.42:
            required_probability = clamp(required_probability + 0.02, 0.35, 0.90)
            size_multiplier = clamp(size_multiplier * 0.90, 0.0, 1.0)
        strong_macro_misalignment = bool(
            trend_like_setup
            and bias_direction in {"bullish", "bearish"}
            and bias_confidence >= 0.70
            and source_confidence >= 0.60
            and authenticity_risk <= 0.35
            and dxy_support_score < 0.45
            and not (
                (bias_direction == "bullish" and candidate.side == "BUY")
                or (bias_direction == "bearish" and candidate.side == "SELL")
            )
        )
        rule_confluence_score = float(risk_summary.get("rule_confluence_score", 0.0))
        rule_confluence_required = float(risk_summary.get("rule_confluence_required", 5.0))
        remote_rejected = False

        fallback_pass = False
        conservative = (
            tier_decision.approved
            and effective_probability >= required_probability
            and spread_points <= max_spread_points
            and adaptive_guard
            and (not remote_rejected)
        )
        narrow_fail = (required_probability - effective_probability) > 0 and (required_probability - effective_probability) <= 0.02
        if (not conservative) and narrow_fail and rule_confluence_score >= rule_confluence_required:
            conservative = True
            size_multiplier = clamp(size_multiplier * 0.6, 0.1, 1.0)
            fallback_pass = True

        micro_mode = bool(risk_summary.get("micro_mode", False))
        if micro_mode and fallback_pass:
            size_multiplier = min(size_multiplier, 0.25)

        if strong_macro_misalignment:
            return False, {
                "approve": False,
                "confidence": effective_probability,
                "probability": effective_probability,
                "recommended_sl_atr_mult": sl_atr_mult,
                "recommended_tp_r": tp_r,
                "trailing_enabled": False,
                "partial_close_enabled": False,
                "max_slippage_points": None,
                "size_multiplier": 0.0,
                "rationale": "final_gate_macro_alignment_reject",
                "ai_mode": ai_mode,
                "remote_fail_open": remote_fail_open,
            }
        if not conservative:
            reason_suffix = adaptive["reason"]
            return False, {
                "approve": False,
                "confidence": effective_probability,
                "probability": effective_probability,
                "recommended_sl_atr_mult": sl_atr_mult,
                "recommended_tp_r": tp_r,
                "trailing_enabled": False,
                "partial_close_enabled": False,
                "max_slippage_points": None,
                "size_multiplier": 0.0,
                "rationale": f"final_gate_conservative_reject:{reason_suffix}",
                "ai_mode": ai_mode,
                "remote_fail_open": remote_fail_open,
            }

        trailing_enabled = regime != "VOLATILE"
        partial_close_enabled = False
        trail_mode = str(remote_advice.get("trail_mode", "ATR")).upper() if remote_advice else "ATR"
        if trail_mode == "NONE":
            # No loosening allowed. Keep deterministic trailing safety.
            trail_mode = "ATR"
        if float(risk_summary.get("rolling_drawdown_pct", 0.0)) >= 0.03:
            trailing_enabled = True

        max_slippage_override: int | None = None
        baseline_slippage = risk_summary.get("max_slippage_points")
        if baseline_slippage is not None:
            reduction = 0.75 if spread_points > (max_spread_points * 0.75) else 0.85 if spread_points > 0 else 1.0
            max_slippage_override = max(1, int(float(baseline_slippage) * reduction))

        rationale = (
            f"final_ok p={effective_probability:.2f} regime={regime} "
            f"news={news_summary.get('source', 'unknown')} bias={bias_direction} "
            f"src={source_confidence:.2f} auth={authenticity_risk:.2f} "
            f"fill={slippage_quality_score:.2f}/{execution_quality_state.lower()} "
            f"hour={hour_expectancy_score:.2f} "
            f"open={len(open_positions)} adapt={adaptive['reason']}"
        )
        if fallback_pass:
            rationale = f"{rationale} fallback_pass=true"
        decision = {
            "approve": True,
            "confidence": effective_probability,
            "probability": effective_probability,
            "recommended_sl_atr_mult": sl_atr_mult,
            "recommended_tp_r": tp_r,
            "trailing_enabled": trailing_enabled,
            "partial_close_enabled": partial_close_enabled,
            "max_slippage_points": max_slippage_override,
            "size_multiplier": size_multiplier,
            "rationale": rationale,
            "fallback_pass": fallback_pass,
            "trail_mode": trail_mode,
            "trail_atr_mult": trail_mult,
            "break_even_r": remote_advice.get("break_even_r") if remote_advice else None,
            "partial_close_r": remote_advice.get("partial_close_r") if remote_advice else None,
            "ai_mode": ai_mode,
            "remote_fail_open": remote_fail_open,
        }
        return True, decision

    def management_advice(
        self,
        position: dict[str, Any],
        features: pd.Series,
        regime: str,
        session_name: str,
        news_summary: dict[str, Any],
        learning_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime_features = features.to_dict() if hasattr(features, "to_dict") else {}
        plan = build_local_management_plan(
            {
                "symbol": str(position.get("symbol", "")),
                "side": str(position.get("side", "")),
                "setup": str(position.get("setup", "")),
                "session": session_name,
                "pnl_r": float(position.get("pnl_r", 0.0) or 0.0),
                "age_minutes": float(position.get("age_minutes", 0.0) or 0.0),
                "spread_points": float(runtime_features.get("m5_spread", 0.0) or 0.0),
                "typical_spread_points": float(runtime_features.get("typical_spread_points", runtime_features.get("m5_spread", 1.0)) or 1.0),
                "runtime_features": runtime_features,
                "learning_brain_bundle": dict((learning_policy or {}).get("bundle") or {}),
            }
        )
        management = plan.get("management_plan", {}) if isinstance(plan.get("management_plan"), dict) else {}
        close_now = bool(str(plan.get("decision", "HOLD")).upper() == "CLOSE")
        if (not news_summary.get("safe", True)) and position.get("pnl_r", 0.0) > 0:
            close_now = True
        return {
            "close_now": close_now,
            "trail_mode": str(management.get("trail_method", "ATR")).upper(),
            "trail_atr_mult": management.get("trail_value"),
            "break_even_r": management.get("move_sl_to_be_at_r"),
            "partial_close_r": management.get("take_partial_at_r"),
            "reason": "local_brain_management",
        }

    def propose_trade_plan(self, context: dict[str, Any]) -> tuple[dict[str, Any], str]:
        return build_local_trade_plan(context), "local_brain"

    def propose_management_plan(self, context: dict[str, Any]) -> tuple[dict[str, Any], str]:
        return build_local_management_plan(context), "local_brain"

    @staticmethod
    def _adaptive_overrides(candidate: SignalCandidate, risk_summary: dict[str, Any]) -> dict[str, Any]:
        samples = float(risk_summary.get("adaptive_samples", 0.0))
        weighted_win_rate = float(risk_summary.get("adaptive_win_rate", 0.5))
        weighted_avg_r = float(risk_summary.get("adaptive_avg_r", 0.0))
        recent_loss_streak = int(float(risk_summary.get("adaptive_recent_loss_streak", 0.0)))

        probability_delta = 0.0
        size_multiplier = 1.0
        reason_parts = ["neutral"]
        if samples >= 6:
            if weighted_win_rate < 0.45:
                probability_delta += min(0.10, (0.45 - weighted_win_rate) * 0.5 + 0.02)
                size_multiplier *= 0.8
                reason_parts = [f"wr_low:{weighted_win_rate:.2f}"]
            elif weighted_win_rate > 0.62 and weighted_avg_r > 0.15:
                probability_delta -= 0.02
                reason_parts = [f"wr_high:{weighted_win_rate:.2f}"]
        if samples >= 8 and weighted_avg_r < 0.0:
            probability_delta += 0.03
            size_multiplier *= 0.85
            reason_parts.append(f"avg_r:{weighted_avg_r:.2f}")
        if recent_loss_streak >= 2:
            probability_delta += 0.03
            size_multiplier *= 0.75
            reason_parts.append(f"loss_streak:{recent_loss_streak}")
        if candidate.setup == "RANGE_REVERSAL" and weighted_win_rate < 0.5 and samples >= 6:
            probability_delta += 0.02
            size_multiplier *= 0.85
            reason_parts.append("range_penalty")
        return {
            "probability_delta": probability_delta,
            "size_multiplier": clamp(size_multiplier, 0.2, 1.0),
            "reason": ",".join(reason_parts),
        }

    def _score_trade(self, candidate: SignalCandidate, row: pd.Series, regime: str) -> tuple[float, float, str]:
        if bool(self.remote_score_enabled) and self.live_remote_enabled:
            remote_probability, remote_ev = self._score_trade_remote(candidate, row, regime)
            if remote_probability is not None:
                return remote_probability, remote_ev, "remote_llm_scorer"

        if self._scorer is not None:
            try:
                vector = self._vectorize(row)
                if hasattr(self._scorer, "predict_proba"):
                    probability = float(self._scorer.predict_proba(vector)[0][1])
                else:
                    probability = float(clamp(self._scorer.predict(vector)[0], 0.0, 1.0))
                if self._value_model is not None:
                    expected_value = float(self._value_model.predict(vector)[0])
                else:
                    expected_value = (probability * candidate.tp_r) - (1.0 - probability)
                return probability, expected_value, "local_model"
            except Exception:
                pass

        trend_strength = abs(float(row.get("h1_trend_score", 0.0))) + abs(float(row.get("m15_trend_score", 0.0)))
        momentum = abs(float(row.get("m5_macd_hist_slope", 0.0))) * 5
        volatility_penalty = max(0.0, float(row.get("m5_atr_pct_of_avg", 1.0)) - 1.3) * 0.08
        base = candidate.score_hint + (trend_strength * 0.1) + momentum - volatility_penalty
        probability = clamp(base, 0.45, 0.82)
        expected_value = (probability * candidate.tp_r) - (1.0 - probability)
        return probability, expected_value, "local_heuristic"

    def _score_trade_remote(self, candidate: SignalCandidate, row: pd.Series, regime: str) -> tuple[float | None, float]:
        if not self.live_remote_enabled:
            return None, 0.0
        context = {
            "symbol": str(row.get("symbol", "")),
            "setup": candidate.setup,
            "candidate_side": candidate.side,
            "regime": regime,
            "spread_points": float(row.get("m5_spread", 0.0)),
            "atr_ratio": float(row.get("m5_atr_pct_of_avg", 1.0)),
            "trend_flag": bool(float(row.get("h1_trend_score", 0.0)) > 0.0),
            "confluence_score": float(candidate.confluence_score),
            "entry_hint": float(row.get("m5_close", 0.0)),
            "stop_atr": float(candidate.stop_atr),
            "tp_r": float(candidate.tp_r),
            "timeframe": "M5",
            "session": "UNKNOWN",
            "news": "unknown",
        }
        payload, _ = self._openai.score_trade(context)
        if payload is None:
            return None, 0.0
        parsed, error = self._parse_remote_trade_payload(payload)
        if parsed is None or error:
            return None, 0.0
        confidence = clamp(float(parsed["confidence"]), 0.0, 1.0)
        tp_r = parsed["risk_adjustment"].get("tp_r")
        expected_value = (confidence * float(tp_r if tp_r is not None else candidate.tp_r)) - (1.0 - confidence)
        return confidence, expected_value

    def _modulate_risk(
        self,
        candidate: SignalCandidate,
        row: pd.Series,
        regime: str,
        consecutive_losses: int,
        probability: float,
    ) -> tuple[float, float, float]:
        if self._risk_model is not None:
            try:
                vector = self._vectorize(row)
                raw = self._risk_model.predict(vector)[0]
                if isinstance(raw, (list, tuple)) and len(raw) >= 3:
                    size_multiplier = clamp(float(raw[0]), 0.25, 1.0)
                    sl_multiplier = clamp(float(raw[1]), 0.8, 1.2)
                    tp_r = clamp(float(raw[2]), 1.5, 2.5)
                    return size_multiplier, sl_multiplier, tp_r
                size_multiplier = clamp(float(raw), 0.25, 1.0)
                return size_multiplier, 1.0, clamp(candidate.tp_r, 1.5, 2.5)
            except Exception:
                pass

        size_multiplier = 1.0
        if regime == "VOLATILE":
            size_multiplier *= 0.7
        elif regime == "RANGING" and candidate.setup == "RANGE_REVERSAL":
            size_multiplier *= 0.9
        if consecutive_losses >= 2:
            size_multiplier *= 0.5
        if probability > 0.7:
            size_multiplier *= 1.05

        sl_multiplier = 1.0
        if regime == "VOLATILE":
            sl_multiplier = 1.1
        elif candidate.setup == "RANGE_REVERSAL":
            sl_multiplier = 0.9

        tp_r = clamp(candidate.tp_r + max(0.0, probability - 0.58), 1.5, 2.5)
        return clamp(size_multiplier, 0.25, 1.0), clamp(sl_multiplier, 0.8, 1.2), tp_r

    def _vectorize(self, row: pd.Series) -> pd.DataFrame:
        columns = self._schema or [column for column in row.index if column != "time"]
        return pd.DataFrame([{column: float(row.get(column, 0.0)) for column in columns}])

    def _fallback_final_order(
        self,
        candidate: SignalCandidate,
        row: pd.Series,
        regime: str,
        tier_decision: AIGateDecision,
        open_positions: list[dict[str, Any]],
        risk_summary: dict[str, Any],
    ) -> tuple[float, float, float]:
        sl_atr_mult = clamp(candidate.stop_atr * tier_decision.sl_multiplier, 0.9, 1.9)
        tp_r = clamp(tier_decision.tp_r, 1.5, 2.25 if regime == "VOLATILE" else 2.5)
        size_multiplier = clamp(tier_decision.size_multiplier, 0.25, 1.0)
        trend_bias = abs(float(row.get("h1_trend_score", 0.0))) + abs(float(row.get("m15_trend_score", 0.0)))
        if trend_bias < 0.5 and candidate.setup != "RANGE_REVERSAL":
            size_multiplier *= 0.6
        if len(open_positions) >= 2:
            size_multiplier *= 0.8
        if bool(risk_summary.get("requires_ai_override", False)):
            size_multiplier *= 0.7
        return sl_atr_mult, tp_r, clamp(size_multiplier, 0.1, 1.0)

    def _remote_committee_decision(
        self,
        candidate: SignalCandidate,
        features: pd.Series,
        regime: str,
        risk_summary: dict[str, Any],
        news_summary: dict[str, Any],
        account_state: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not self.live_remote_enabled:
            return None, "live_remote_disabled"
        context = {
            "symbol": str(candidate.signal_id.split("-")[0]).upper(),
            "timeframe": "M5",
            "candidate": {
                "setup": candidate.setup,
                "direction": self._candidate_direction(candidate.side),
                "entry_hint": float(features.get("m5_close", 0.0)),
                "stop_atr": float(candidate.stop_atr),
                "tp_r": float(candidate.tp_r),
                "confluence_score": float(candidate.confluence_score),
                "confluence_required": float(candidate.confluence_required),
            },
            "regime": regime,
            "regime_flags": {
                "trend": bool(features.get("h1_trend_score", 0.0)),
                "atr_ratio": float(features.get("m5_atr_pct_of_avg", 1.0)),
            },
            "session": str(risk_summary.get("session_name", "UNKNOWN")),
            "spread_points": float(risk_summary.get("spread_points", 0.0)),
            "news": {
                "safe": bool(news_summary.get("safe", False)),
                "reason": str(news_summary.get("reason", "")),
                "source": str(news_summary.get("source", "")),
                "bias_direction": str(news_summary.get("bias_direction", "neutral")),
                "bias_confidence": float(news_summary.get("bias_confidence", 0.0)),
                "source_confidence": float(news_summary.get("source_confidence", 0.5)),
                "authenticity_risk": float(news_summary.get("authenticity_risk", 0.0)),
                "sentiment_extreme": float(news_summary.get("sentiment_extreme", 0.0)),
                "crowding_bias": str(news_summary.get("crowding_bias", "neutral")),
            },
            "account": {
                "equity": float(account_state.get("equity", 0.0)),
            },
        }
        payload, error = self._openai.score_trade(context)
        if payload is None:
            return None, error
        parsed, parse_error = self._parse_remote_trade_payload(payload)
        if parsed is None:
            return None, parse_error or "remote_schema_invalid"
        return parsed, None

    def _remote_management_decision(
        self,
        position: dict[str, Any],
        features: pd.Series,
        regime: str,
        session_name: str,
        news_summary: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not self.live_remote_enabled:
            return None, "live_remote_disabled"
        context = {
            "position": {
                "signal_id": str(position.get("signal_id", "")),
                "symbol": str(position.get("symbol", "")),
                "side": str(position.get("side", "")),
                "entry_price": float(position.get("entry_price", 0.0)),
                "sl": float(position.get("sl", 0.0)),
                "tp": float(position.get("tp", 0.0)),
                "pnl_r": float(position.get("pnl_r", 0.0)),
            },
            "market": {
                "price": float(features.get("m5_close", 0.0)),
                "spread_points": float(features.get("m5_spread", 0.0)),
                "atr": float(features.get("m5_atr_14", 0.0)),
            },
            "regime": regime,
            "session": session_name,
            "news": {
                "safe": bool(news_summary.get("safe", False)),
                "reason": str(news_summary.get("reason", "")),
            },
        }
        payload, error = self._openai.management_advice(context)
        if payload is None:
            return None, error
        if not isinstance(payload, dict):
            return None, "management_schema_invalid"
        return payload, None

    @staticmethod
    def _parse_trade_plan_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        decision = str(payload.get("decision", "PASS")).upper()
        if decision not in {"TAKE", "PASS"}:
            return None
        setup_type = str(payload.get("setup_type", "scalp")).lower()
        if setup_type not in {"scalp", "daytrade", "grid_manage"}:
            return None
        side = str(payload.get("side", "")).upper()
        if side not in {"BUY", "SELL", ""}:
            return None
        risk_tier = str(payload.get("risk_tier", "NORMAL")).upper()
        if risk_tier not in {"LOW", "NORMAL", "HIGH"}:
            risk_tier = "NORMAL"
        management = payload.get("management_plan", {})
        if not isinstance(management, dict):
            management = {}
        return {
            "decision": decision,
            "setup_type": setup_type,
            "side": side,
            "sl_points": float(payload.get("sl_points", 0.0) or 0.0),
            "tp_points": float(payload.get("tp_points", 0.0) or 0.0),
            "rr_target": float(payload.get("rr_target", 0.0) or 0.0),
            "confidence": clamp(float(payload.get("confidence", 0.0) or 0.0), 0.0, 1.0),
            "expected_value_r": float(payload.get("expected_value_r", 0.0) or 0.0),
            "risk_tier": risk_tier,
            "management_plan": {
                "move_sl_to_be_at_r": management.get("move_sl_to_be_at_r"),
                "trail_after_r": management.get("trail_after_r"),
                "trail_method": str(management.get("trail_method", "atr")).lower(),
                "trail_value": management.get("trail_value"),
                "take_partial_at_r": management.get("take_partial_at_r"),
                "time_stop_minutes": management.get("time_stop_minutes"),
                "early_exit_rules": str(management.get("early_exit_rules", "")),
            },
            "notes": str(payload.get("notes", "")),
        }

    @staticmethod
    def _parse_management_plan_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        decision = str(payload.get("decision", "HOLD")).upper()
        if decision not in {"HOLD", "MODIFY", "CLOSE"}:
            return None
        management = payload.get("management_plan", {})
        if not isinstance(management, dict):
            management = {}
        return {
            "decision": decision,
            "confidence": clamp(float(payload.get("confidence", 0.0) or 0.0), 0.0, 1.0),
            "management_plan": {
                "move_sl_to_be_at_r": management.get("move_sl_to_be_at_r"),
                "trail_after_r": management.get("trail_after_r"),
                "trail_method": str(management.get("trail_method", "atr")).lower(),
                "trail_value": management.get("trail_value"),
                "take_partial_at_r": management.get("take_partial_at_r"),
                "time_stop_minutes": management.get("time_stop_minutes"),
                "early_exit_rules": str(management.get("early_exit_rules", "")),
            },
            "notes": str(payload.get("notes", "")),
        }

    @staticmethod
    def _fallback_trade_plan(context: dict[str, Any]) -> dict[str, Any]:
        setup = str(context.get("setup", "")).upper()
        setup_type = "daytrade" if any(token in setup for token in ("DAY", "SET_FORGET", "H1", "H4")) else "scalp"
        if "GRID" in setup:
            setup_type = "grid_manage"
        side = str(context.get("side", "BUY")).upper()
        spread = max(0.0, float(context.get("spread_points", 0.0)))
        point = max(1e-9, float(context.get("point_size", 0.0001)))
        atr_price = max(point, float(context.get("atr_price", point * 40.0)))
        atr_points = max(1.0, atr_price / point)
        sl_points = max(atr_points * (1.0 if setup_type == "scalp" else 1.4), float(context.get("min_stop_points", 10.0)))
        rr_target = 1.9 if setup_type == "daytrade" else 1.4
        tp_points = sl_points * rr_target
        risk_tier = "LOW" if setup_type != "daytrade" else "NORMAL"
        confidence = clamp(float(context.get("probability", 0.6)) - min(0.10, spread / 1000.0), 0.3, 0.95)
        decision = "TAKE" if confidence >= float(context.get("min_probability", 0.55)) else "PASS"
        return {
            "decision": decision,
            "setup_type": setup_type,
            "side": side,
            "sl_points": sl_points,
            "tp_points": tp_points,
            "rr_target": rr_target,
            "confidence": confidence,
            "expected_value_r": float(context.get("expected_value_r", 0.2)),
            "risk_tier": risk_tier,
            "management_plan": {
                "move_sl_to_be_at_r": 0.8 if setup_type == "scalp" else 1.2,
                "trail_after_r": 1.0 if setup_type == "scalp" else 1.5,
                "trail_method": "atr",
                "trail_value": 1.0 if setup_type == "scalp" else 1.4,
                "take_partial_at_r": 1.0 if setup_type == "scalp" else 2.0,
                "time_stop_minutes": 45 if setup_type == "scalp" else 240,
                "early_exit_rules": "exit_on_news_or_momentum_stall",
            },
            "notes": "fallback_trade_plan",
        }

    @staticmethod
    def _fallback_management_plan(context: dict[str, Any]) -> dict[str, Any]:
        pnl_r = float(context.get("pnl_r", 0.0))
        decision = "HOLD"
        if pnl_r >= 1.0:
            decision = "MODIFY"
        if pnl_r <= -1.2:
            decision = "CLOSE"
        return {
            "decision": decision,
            "confidence": 0.65,
            "management_plan": {
                "move_sl_to_be_at_r": 0.8 if pnl_r > 0 else None,
                "trail_after_r": 1.0,
                "trail_method": "atr",
                "trail_value": 1.0,
                "take_partial_at_r": 1.0,
                "time_stop_minutes": 60,
                "early_exit_rules": "close_if_stall_or_adverse_news",
            },
            "notes": "fallback_management_plan",
        }

    @staticmethod
    def _parse_remote_trade_payload(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(payload, dict):
            return None, "payload_not_object"
        if not isinstance(payload.get("approve"), bool):
            return None, "missing_approve"
        if "confidence" not in payload:
            return None, "missing_confidence"
        direction = str(payload.get("direction", "NONE")).upper()
        if direction not in {"LONG", "SHORT", "NONE"}:
            return None, "invalid_direction"
        reasons = payload.get("reasons", [])
        if not isinstance(reasons, list):
            return None, "invalid_reasons"
        risk_adjustment = payload.get("risk_adjustment", {})
        if not isinstance(risk_adjustment, dict):
            return None, "invalid_risk_adjustment"
        trail_mode = str(risk_adjustment.get("trail_mode", "ATR")).upper()
        if trail_mode not in {"ATR", "STRUCTURE", "NONE"}:
            return None, "invalid_trail_mode"

        parsed = {
            "approve": bool(payload["approve"]),
            "confidence": clamp(float(payload.get("confidence", 0.0)), 0.0, 1.0),
            "direction": direction,
            "reasons": [str(item) for item in reasons],
            "risk_adjustment": {
                "size_multiplier": clamp(float(risk_adjustment.get("size_multiplier", 1.0)), 0.1, 1.0),
                "tp_r": float(risk_adjustment["tp_r"]) if risk_adjustment.get("tp_r") is not None else None,
                "trail_mode": trail_mode,
                "trail_atr_mult": float(risk_adjustment["trail_atr_mult"]) if risk_adjustment.get("trail_atr_mult") is not None else None,
                "break_even_r": float(risk_adjustment["break_even_r"]) if risk_adjustment.get("break_even_r") is not None else None,
                "partial_close_r": float(risk_adjustment["partial_close_r"]) if risk_adjustment.get("partial_close_r") is not None else None,
            },
        }
        return parsed, None

    @staticmethod
    def _candidate_direction(side: str) -> str:
        return "LONG" if str(side).upper() == "BUY" else "SHORT"

    def _load_schema(self) -> list[str]:
        if not self.schema_path.exists():
            return []
        try:
            payload = json.loads(self.schema_path.read_text())
            return [str(item) for item in payload.get("features", [])]
        except Exception:
            return []

    @staticmethod
    def _safe_load(path: Path):
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None
