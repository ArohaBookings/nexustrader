from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import json
from zoneinfo import ZoneInfo

import pandas as pd

from src.aggression_runtime import score_shadow_variant
from src.lane_governor import (
    build_loss_attribution_summary,
    build_portable_funded_profile,
    build_shadow_challenger_pool,
    build_walk_forward_scorecards,
    evaluate_execution_quality_gate,
    resolve_lane_lifecycle,
)
from src.online_learning import OnlineLearningEngine
from src.openai_client import OpenAIClient
from src.strategy_optimizer import StrategyOptimizer
from src.utils import clamp, ensure_parent, utc_now


SYDNEY = ZoneInfo("Australia/Sydney")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _symbol_key(value: Any) -> str:
    return "".join(char for char in str(value or "").upper() if char.isalnum())


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return float((ordered[mid - 1] + ordered[mid]) / 2.0)
    return float(ordered[mid])


def _hour_quality_tier(*, expectancy_r: float, win_rate: float, trade_count: int) -> str:
    if trade_count >= 3 and expectancy_r >= 0.50 and win_rate >= 0.72:
        return "A+"
    if trade_count >= 2 and expectancy_r >= 0.20 and win_rate >= 0.60:
        return "A"
    if expectancy_r >= 0.05 and win_rate >= 0.52:
        return "B"
    if expectancy_r > -0.05 and win_rate >= 0.45:
        return "B-"
    return "C"


@dataclass(frozen=True)
class OfflineGPTReview:
    generated_at: str
    status: str
    summary: str = ""
    weak_patterns: list[str] = field(default_factory=list)
    strategy_ideas: list[str] = field(default_factory=list)
    next_cycle_focus: list[str] = field(default_factory=list)
    reentry_watchlist: list[str] = field(default_factory=list)
    weekly_trade_ideas: list[str] = field(default_factory=list)
    hybrid_pair_ideas: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True)
class LearningPromotionBundle:
    generated_at: str
    ga_generation_id: int
    mode: str
    monte_carlo_min_realities: int
    monte_carlo_pass_floor: float
    risk_pct_target: float
    max_open_trades_multiplier: float
    lot_size_multiplier: float
    quota_catchup_pressure: float
    shadow_pair_focus: list[str]
    weak_pair_focus: list[str]
    promoted_patterns: list[str]
    trajectory_projection: dict[str, float]
    goal_state: dict[str, Any] = field(default_factory=dict)
    self_heal_actions: list[dict[str, Any]] = field(default_factory=list)
    shadow_strategy_variants: list[dict[str, Any]] = field(default_factory=list)
    risk_reduction_active: bool = False
    recovery_mode_active: bool = False
    pair_directives: dict[str, dict[str, Any]] = field(default_factory=dict)
    meeting_packet: dict[str, Any] = field(default_factory=dict)
    local_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrainCycleReport:
    generated_at: str
    session_name: str
    local_cycle_status: str
    offline_gpt_status: str
    weekly_prep_status: str
    trades_reviewed: int
    projection: dict[str, float]
    promotion_bundle: LearningPromotionBundle
    offline_gpt_review: OfflineGPTReview | None = None
    notes: list[str] = field(default_factory=list)


class ApexLearningBrain:
    def __init__(
        self,
        *,
        online_learning: OnlineLearningEngine,
        strategy_optimizer: StrategyOptimizer,
        journal: Any,
        data_dir: Path,
        logger: Any | None = None,
        offline_gpt_enabled: bool = True,
        openai_api_env: str = "OPENAI_API_KEY",
        openai_model: str = "gpt-4o-mini",
        openai_timeout_seconds: float = 8.0,
        openai_retry_once: bool = False,
        last_trades_review_limit: int = 200,
        monte_carlo_min_realities: int = 500,
        monte_carlo_pass_floor: float = 0.88,
        new_pair_observation_days: int = 1,
        new_pair_observation_hours: int = 4,
        short_goal_equity: float = 100000.0,
        medium_goal_equity: float = 1000000.0,
        shadow_default_variants: int = 50,
        shadow_hot_variants: int = 64,
        shadow_promotion_threshold: float = 0.64,
    ) -> None:
        self.online_learning = online_learning
        self.strategy_optimizer = strategy_optimizer
        self.journal = journal
        self.data_dir = Path(data_dir)
        self.logger = logger
        self.last_trades_review_limit = max(20, int(last_trades_review_limit))
        self.monte_carlo_min_realities = max(500, int(monte_carlo_min_realities))
        self.monte_carlo_pass_floor = clamp(float(monte_carlo_pass_floor), 0.5, 0.99)
        self.new_pair_observation_days = max(1, int(new_pair_observation_days))
        self.new_pair_observation_hours = max(1, int(new_pair_observation_hours))
        self.short_goal_equity = max(1000.0, float(short_goal_equity))
        self.medium_goal_equity = max(float(self.short_goal_equity), float(medium_goal_equity))
        self.shadow_default_variants = max(4, int(shadow_default_variants))
        self.shadow_hot_variants = max(self.shadow_default_variants, int(shadow_hot_variants))
        self.shadow_promotion_threshold = clamp(float(shadow_promotion_threshold), 0.40, 0.95)
        self.brain_state_path = self.data_dir / "apex_learning_brain_state.json"
        self.promotion_bundle_path = self.data_dir / "apex_learning_promotions.json"
        self.portable_profile_path = self.data_dir / "apex_portable_funded_profile.json"
        self.cycle_log_path = self.data_dir / "apex_learning_brain_log.jsonl"
        self.weekly_ideas_path = self.data_dir / "apex_weekly_ideas.json"
        self.offline_review_path = self.data_dir / "apex_offline_gpt_review.json"
        self.live_feedback_path = self.data_dir / "apex_learning_live_feedback.jsonl"
        self._offline_gpt = OpenAIClient(
            api_key_env=openai_api_env,
            model=openai_model,
            timeout_seconds=float(openai_timeout_seconds),
            retry_once=bool(openai_retry_once),
            enabled=bool(offline_gpt_enabled),
            logger=logger,
        )
        self._last_report: BrainCycleReport | None = None
        self._last_promotion_bundle: LearningPromotionBundle = self._load_promotion_bundle()
        self._brain_state = self._load_json(self.brain_state_path, {})

    def run_cycle(
        self,
        *,
        now_utc: datetime | None = None,
        session_name: str = "",
        account_state: dict[str, Any] | None = None,
        runtime_state: dict[str, Any] | None = None,
        weekly_prep: bool = False,
        force_local_retrain: bool = False,
    ) -> BrainCycleReport:
        now = now_utc or utc_now()
        account_state = dict(account_state or {})
        runtime_state = dict(runtime_state or {})
        current_session_name = str(
            session_name or runtime_state.get("current_session_name") or runtime_state.get("session_name") or ""
        ).upper()
        local_memory = self._load_local_memory(limit=self.last_trades_review_limit)
        local_summary = self._build_local_summary(local_memory)
        self.online_learning.maybe_retrain_maintenance(
            now_utc=now,
            session_name=str(session_name or "").upper(),
            active_sessions={"TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            force=bool(force_local_retrain),
        )
        learning_status = self.online_learning.status_snapshot()
        optimizer_summary = self.strategy_optimizer.summary()
        projection = self._trajectory_projection(
            local_summary=local_summary,
            account_state=account_state,
        )
        goal_state = self._goal_state(projection=projection)
        anomaly_state = self._build_autonomy_state(
            now=now,
            local_summary=local_summary,
            learning_status=learning_status,
            runtime_state=runtime_state,
            account_state=account_state,
            projection=projection,
            goal_state=goal_state,
        )
        shadow_strategy_variants = self._build_shadow_strategy_variants(
            now=now,
            local_summary=local_summary,
            optimizer_summary=optimizer_summary,
            runtime_state=runtime_state,
        )
        pair_directives = self._build_pair_directives(
            local_summary=local_summary,
            runtime_state=runtime_state,
            learning_status=learning_status,
            goal_state=goal_state,
            current_session_name=current_session_name,
            shadow_strategy_variants=shadow_strategy_variants,
        )
        meeting_packet = self._build_meeting_packet(
            now=now,
            local_summary=local_summary,
            learning_status=learning_status,
            projection=projection,
            goal_state=goal_state,
            runtime_state=runtime_state,
            pair_directives=pair_directives,
            current_session_name=current_session_name,
        )
        initial_promotion_bundle = self._build_promotion_bundle(
            now=now,
            local_summary=local_summary,
            learning_status=learning_status,
            optimizer_summary=optimizer_summary,
            projection=projection,
            goal_state=goal_state,
            anomaly_state=anomaly_state,
            shadow_strategy_variants=shadow_strategy_variants,
            runtime_state=runtime_state,
            pair_directives=pair_directives,
            meeting_packet=meeting_packet,
        )
        offline_review = self._offline_gpt_review(
            now=now,
            weekly_prep=bool(weekly_prep),
            local_summary=local_summary,
            promotion_bundle=initial_promotion_bundle,
            projection=projection,
            learning_status=learning_status,
        )
        pair_directives = self._apply_gpt_hybrid_pair_ideas(
            pair_directives=pair_directives,
            offline_review=offline_review,
            current_session_name=current_session_name,
        )
        meeting_packet = self._build_meeting_packet(
            now=now,
            local_summary=local_summary,
            learning_status=learning_status,
            projection=projection,
            goal_state=goal_state,
            runtime_state=runtime_state,
            pair_directives=pair_directives,
            current_session_name=current_session_name,
            offline_review=offline_review,
        )
        promotion_bundle = self._build_promotion_bundle(
            now=now,
            local_summary=local_summary,
            learning_status=learning_status,
            optimizer_summary=optimizer_summary,
            projection=projection,
            goal_state=goal_state,
            anomaly_state=anomaly_state,
            shadow_strategy_variants=shadow_strategy_variants,
            runtime_state=runtime_state,
            pair_directives=pair_directives,
            meeting_packet=meeting_packet,
        )
        self._last_promotion_bundle = promotion_bundle
        self._write_json(self.promotion_bundle_path, asdict(promotion_bundle))
        portable_profile = build_portable_funded_profile(
            generated_at=now,
            pair_directives=pair_directives,
            meeting_packet=meeting_packet,
            local_summary=local_summary,
            shadow_strategy_variants=shadow_strategy_variants,
        )
        self._write_json(self.portable_profile_path, portable_profile)
        if weekly_prep:
            self._write_json(
                self.weekly_ideas_path,
                {
                    "generated_at": now.isoformat(),
                    "status": str(offline_review.status),
                    "weekly_trade_ideas": list(offline_review.weekly_trade_ideas),
                    "reentry_watchlist": list(offline_review.reentry_watchlist),
                    "summary": str(offline_review.summary),
                },
            )
        self._write_json(self.offline_review_path, asdict(offline_review))
        report = BrainCycleReport(
            generated_at=now.isoformat(),
            session_name=str(session_name or "").upper(),
            local_cycle_status=str(learning_status.get("last_maintenance_status") or "ok"),
            offline_gpt_status=str(offline_review.status),
            weekly_prep_status="generated" if weekly_prep else "idle",
            trades_reviewed=int(local_summary.get("reviewed_trade_count", 0)),
            projection=projection,
            promotion_bundle=promotion_bundle,
            offline_gpt_review=offline_review,
            notes=[
                f"gpt_suggestions={len(list(offline_review.strategy_ideas)) + len(list(offline_review.hybrid_pair_ideas))}",
                f"promoted_patterns={len(list(promotion_bundle.promoted_patterns))}",
            ],
        )
        self._last_report = report
        self._brain_state = {
            "last_report": asdict(report),
            "last_local_cycle_at": report.generated_at,
            "last_local_cycle_status": report.local_cycle_status,
            "last_offline_gpt_review_at": report.generated_at,
            "last_offline_gpt_review_status": report.offline_gpt_status,
            "last_weekly_prep_at": report.generated_at if weekly_prep else str(self._brain_state.get("last_weekly_prep_at") or ""),
            "last_weekly_prep_status": report.weekly_prep_status if weekly_prep else str(self._brain_state.get("last_weekly_prep_status") or "idle"),
            "trajectory_projection": projection,
            "goal_state": goal_state,
            "autonomy_state": anomaly_state,
            "self_heal_failure_counts": dict(anomaly_state.get("failure_counts") or {}),
            "last_self_heal_actions": list(promotion_bundle.self_heal_actions),
            "last_shadow_strategy_variants": list(promotion_bundle.shadow_strategy_variants),
            "risk_reduction_active": bool(promotion_bundle.risk_reduction_active),
            "recovery_mode_active": bool(promotion_bundle.recovery_mode_active),
            "pair_directives": dict(promotion_bundle.pair_directives),
            "meeting_packet": dict(promotion_bundle.meeting_packet),
            "portable_funded_profile": dict(portable_profile),
            "last_promotion_summary": promotion_bundle.local_summary,
            "new_pair_observation_summary": local_summary.get("new_pair_observation_summary", {}),
            "gpt_suggestion_count": len(list(offline_review.strategy_ideas))
            + len(list(offline_review.weekly_trade_ideas))
            + len(list(offline_review.hybrid_pair_ideas)),
            "local_promotion_count": len(list(promotion_bundle.promoted_patterns)),
        }
        self._write_json(self.brain_state_path, self._brain_state)
        self._append_jsonl(
            self.cycle_log_path,
            {
                "generated_at": report.generated_at,
                "session_name": report.session_name,
                "local_cycle_status": report.local_cycle_status,
                "offline_gpt_status": report.offline_gpt_status,
                "weekly_prep_status": report.weekly_prep_status,
                "projection": report.projection,
                "goal_state": goal_state,
                "autonomy_state": anomaly_state,
                "local_summary": promotion_bundle.local_summary,
                "self_heal_actions": list(promotion_bundle.self_heal_actions),
                "shadow_strategy_variants": list(promotion_bundle.shadow_strategy_variants),
                "pair_directives": dict(promotion_bundle.pair_directives),
                "meeting_packet": dict(promotion_bundle.meeting_packet),
                "gpt_summary": asdict(offline_review),
            },
        )
        return report

    def status_snapshot(self) -> dict[str, Any]:
        report = self._last_report
        promotion = self._last_promotion_bundle
        return {
            "mode": "local_live_offline_gpt",
            "last_local_cycle_at": str(self._brain_state.get("last_local_cycle_at") or ""),
            "last_local_cycle_status": str(self._brain_state.get("last_local_cycle_status") or ""),
            "last_offline_gpt_review_at": str(self._brain_state.get("last_offline_gpt_review_at") or ""),
            "last_offline_gpt_review_status": str(self._brain_state.get("last_offline_gpt_review_status") or ""),
            "last_weekly_prep_at": str(self._brain_state.get("last_weekly_prep_at") or ""),
            "last_weekly_prep_status": str(self._brain_state.get("last_weekly_prep_status") or "idle"),
            "last_cycle_status": str(self._brain_state.get("last_local_cycle_status") or ""),
            "last_gpt_review_status": str(self._brain_state.get("last_offline_gpt_review_status") or ""),
            "last_live_feedback_at": str(self._brain_state.get("last_live_feedback_at") or ""),
            "live_feedback_count": int(self._brain_state.get("live_feedback_count") or 0),
            "current_ga_generation_id": int(promotion.ga_generation_id),
            "current_ga_generation": int(promotion.ga_generation_id),
            "trajectory_projection": dict(self._brain_state.get("trajectory_projection") or promotion.trajectory_projection),
            "goal_state": dict(self._brain_state.get("goal_state") or promotion.goal_state),
            "autonomy_state": dict(self._brain_state.get("autonomy_state") or {}),
            "self_heal_actions": list(self._brain_state.get("last_self_heal_actions") or promotion.self_heal_actions),
            "shadow_strategy_variants": list(self._brain_state.get("last_shadow_strategy_variants") or promotion.shadow_strategy_variants),
            "risk_reduction_active": bool(self._brain_state.get("risk_reduction_active") or promotion.risk_reduction_active),
            "recovery_mode_active": bool(self._brain_state.get("recovery_mode_active") or promotion.recovery_mode_active),
            "pair_directives": dict(self._brain_state.get("pair_directives") or promotion.pair_directives),
            "meeting_packet": dict(self._brain_state.get("meeting_packet") or promotion.meeting_packet),
            "portable_funded_profile": dict(self._load_json(self.portable_profile_path, {})),
            "last_promotion_summary": dict(self._brain_state.get("last_promotion_summary") or promotion.local_summary),
            "new_pair_observation_summary": dict(self._brain_state.get("new_pair_observation_summary") or {}),
            "gpt_suggestion_count": int(self._brain_state.get("gpt_suggestion_count") or 0),
            "local_promotion_count": int(self._brain_state.get("local_promotion_count") or 0),
            "last_trades_review_limit": int(self.last_trades_review_limit),
            "monte_carlo_min_realities": int(self.monte_carlo_min_realities),
            "monte_carlo_pass_floor": float(self.monte_carlo_pass_floor),
            "offline_gpt_health": dict(self._offline_gpt.health()),
            "active_promotion_bundle": asdict(promotion),
            "last_cycle_report": asdict(report) if report is not None else {},
        }

    def promotion_bundle_snapshot(self) -> dict[str, Any]:
        return asdict(self._last_promotion_bundle)

    def live_policy_snapshot(self, *, symbol: str = "", setup: str = "") -> dict[str, Any]:
        weekly_ideas = self._load_json(self.weekly_ideas_path, {})
        symbol_key = str(symbol or "").upper()
        setup_key = str(setup or "").upper()
        weak_pair_focus = {
            str(item).upper()
            for item in self._last_promotion_bundle.weak_pair_focus
            if str(item).strip()
        }
        promoted_patterns = {
            str(item).upper()
            for item in self._last_promotion_bundle.promoted_patterns
            if str(item).strip()
        }
        pair_directive = dict((self._last_promotion_bundle.pair_directives or {}).get(symbol_key) or {})
        hour_expectancy_matrix = dict((self._last_promotion_bundle.meeting_packet or {}).get("hour_expectancy_matrix") or {})
        setup_hour_directive = dict(hour_expectancy_matrix.get(f"{symbol_key}:{setup_key}") or {})
        return {
            "bundle": asdict(self._last_promotion_bundle),
            "trajectory_projection": dict(self._last_promotion_bundle.trajectory_projection),
            "goal_state": dict(self._last_promotion_bundle.goal_state),
            "self_heal_actions": list(self._last_promotion_bundle.self_heal_actions),
            "shadow_strategy_variants": list(self._last_promotion_bundle.shadow_strategy_variants),
            "pair_directives": dict(self._last_promotion_bundle.pair_directives),
            "meeting_packet": dict(self._last_promotion_bundle.meeting_packet),
            "portable_funded_profile": dict(self._load_json(self.portable_profile_path, {})),
            "weekly_trade_ideas": [str(item) for item in weekly_ideas.get("weekly_trade_ideas", []) if str(item).strip()],
            "reentry_watchlist": [str(item) for item in weekly_ideas.get("reentry_watchlist", []) if str(item).strip()],
            "proven_reentry_queue": [
                str(item)
                for item in (self._last_promotion_bundle.local_summary.get("proven_reentry_queue") or [])
                if str(item).strip()
            ],
            "symbol": symbol_key,
            "setup": setup_key,
            "pair_directive": pair_directive,
            "setup_hour_directive": setup_hour_directive,
            "management_directives": dict(pair_directive.get("management_directives") or {}),
            "frequency_directives": dict(pair_directive.get("frequency_directives") or {}),
            "blocker_relaxation_state": dict(pair_directive.get("blocker_relaxation_state") or {}),
            "lane_state_machine": dict(pair_directive.get("lane_state_machine") or {}),
            "walk_forward_scorecards": dict(pair_directive.get("walk_forward_scorecards") or {}),
            "loss_attribution_summary": dict(pair_directive.get("loss_attribution_summary") or {}),
            "shadow_challenger_pool": dict(pair_directive.get("shadow_challenger_pool") or {}),
            "execution_quality_directives": dict(pair_directive.get("execution_quality_directives") or {}),
            "gpt_hybrid_advisory": dict(pair_directive.get("gpt_hybrid_advisory") or {}),
            "trade_horizon_bias": str(pair_directive.get("trade_horizon_bias") or ""),
            "session_focus": list(pair_directive.get("session_focus") or []),
            "aggression_multiplier": float(pair_directive.get("aggression_multiplier", 1.0) or 1.0),
            "min_confluence_override": float(pair_directive.get("min_confluence_override", 0.0) or 0.0),
            "reentry_priority": float(pair_directive.get("reentry_priority", 0.0) or 0.0),
            "symbol_is_weak_focus": bool(symbol_key and symbol_key in weak_pair_focus),
            "setup_is_promoted_pattern": bool(setup_key and setup_key in promoted_patterns),
        }

    def apply_promoted_params_to_runtime(self, runtime_state: dict[str, Any] | None = None) -> dict[str, Any]:
        target = runtime_state if isinstance(runtime_state, dict) else {}
        target["learning_brain_bundle"] = asdict(self._last_promotion_bundle)
        target["learning_brain_projection"] = dict(self._last_promotion_bundle.trajectory_projection)
        target["learning_goal_state"] = dict(self._last_promotion_bundle.goal_state)
        target["learning_brain_self_heal_actions"] = list(self._last_promotion_bundle.self_heal_actions)
        target["learning_brain_shadow_variants"] = list(self._last_promotion_bundle.shadow_strategy_variants)
        target["learning_brain_portable_profile"] = dict(self._load_json(self.portable_profile_path, {}))
        target["learning_brain_status"] = self.status_snapshot()
        return target

    def record_trade_feedback(self, feedback: dict[str, Any]) -> None:
        payload = {
            "logged_at": utc_now().isoformat(),
            **dict(feedback or {}),
        }
        self._append_jsonl(self.live_feedback_path, payload)
        self._brain_state["last_live_feedback_at"] = str(payload["logged_at"])
        self._brain_state["live_feedback_count"] = int(self._brain_state.get("live_feedback_count") or 0) + 1
        self._write_json(self.brain_state_path, self._brain_state)

    def _load_local_memory(self, *, limit: int) -> dict[str, Any]:
        closed_trades = list(self.journal.closed_trades(max(1, int(limit))))
        closed_frame = pd.DataFrame(closed_trades)
        setup_path = getattr(self.online_learning, "setup_log_path", None)
        trades_path = getattr(self.online_learning, "data_path", None)
        setup_frame = (
            pd.read_csv(setup_path, low_memory=False)
            if isinstance(setup_path, Path) and setup_path.exists()
            else pd.DataFrame()
        )
        history_frame = (
            pd.read_csv(trades_path, low_memory=False)
            if isinstance(trades_path, Path) and trades_path.exists()
            else pd.DataFrame()
        )
        live_feedback_rows = self._load_jsonl_rows(self.live_feedback_path, limit=max(50, int(limit) * 6))
        live_feedback_frame = pd.DataFrame(live_feedback_rows)
        return {
            "closed_trades": closed_trades,
            "closed_frame": closed_frame,
            "setup_frame": setup_frame,
            "history_frame": history_frame,
            "live_feedback_rows": live_feedback_rows,
            "live_feedback_frame": live_feedback_frame,
        }

    @staticmethod
    def _load_jsonl_rows(path: Path | None, *, limit: int) -> list[dict[str, Any]]:
        if not isinstance(path, Path) or not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = str(line or "").strip()
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        rows.append(payload)
        except Exception:
            return []
        if limit > 0 and len(rows) > int(limit):
            rows = rows[-int(limit) :]
        return rows

    def _build_local_summary(self, local_memory: dict[str, Any]) -> dict[str, Any]:
        closed_trades = list(local_memory.get("closed_trades") or [])
        live_feedback_rows = list(local_memory.get("live_feedback_rows") or [])
        pnl_r = [_safe_float(row.get("pnl_r"), 0.0) for row in closed_trades]
        wins = [value for value in pnl_r if value >= 0.0]
        losses = [abs(value) for value in pnl_r if value < 0.0]
        winning_setup_counter: Counter[str] = Counter()
        reentry_counter: Counter[str] = Counter()
        failure_counter: Counter[str] = Counter()
        pair_rows: dict[str, list[dict[str, Any]]] = {}
        lane_rows: dict[str, list[dict[str, Any]]] = {}
        for row in closed_trades:
            symbol_key = _symbol_key(row.get("symbol"))
            setup_name = str(row.get("setup") or row.get("strategy_key") or row.get("symbol") or "").strip().upper()
            pnl_value = _safe_float(row.get("pnl_r"), 0.0)
            if not setup_name:
                continue
            if symbol_key:
                pair_rows.setdefault(symbol_key, []).append(dict(row))
                lane_key = f"{symbol_key}:{setup_name}"
                lane_rows.setdefault(lane_key, []).append(dict(row))
            if pnl_value >= 0.0:
                winning_setup_counter[setup_name] += 1
            if pnl_value >= 0.35:
                reentry_counter[setup_name] += 1
            else:
                failure_counter[self._failure_reason(row)] += 1
        expectancy_r = _mean(pnl_r)
        profit_factor = (sum(wins) / sum(losses)) if losses else (float("inf") if wins else 0.0)
        win_rate = (len(wins) / len(pnl_r)) if pnl_r else 0.0
        winner_loss_ratio = (_mean(wins) / _mean(losses)) if wins and losses else (_mean(wins) if wins else 0.0)
        drawdown_penalty = max(0.0, -min(0.0, min(pnl_r) if pnl_r else 0.0))
        history_frame = local_memory.get("history_frame")
        symbol_counts: dict[str, int] = {}
        observation_summary: dict[str, Any] = {}
        if isinstance(history_frame, pd.DataFrame) and not history_frame.empty:
            for symbol, frame in history_frame.groupby(history_frame["symbol"].astype(str).str.upper()):
                symbol_counts[str(symbol)] = int(len(frame))
                try:
                    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce").dropna()
                except Exception:
                    timestamps = pd.Series(dtype="datetime64[ns, UTC]")
                if timestamps.empty:
                    day_span = 0
                    hour_span = 0.0
                else:
                    span = timestamps.max() - timestamps.min()
                    day_span = int(max(0, span.days)) + 1
                    hour_span = max(0.0, float(span.total_seconds()) / 3600.0)
                ready_for_generation = bool(
                    day_span >= self.new_pair_observation_days
                    or hour_span >= float(self.new_pair_observation_hours)
                )
                observation_summary[str(symbol)] = {
                    "trade_count": int(len(frame)),
                    "observation_days": int(day_span),
                    "observation_hours": float(round(hour_span, 2)),
                    "ready_for_local_strategy_generation": ready_for_generation,
                }
        pair_stats = {
            symbol: self._pair_stats(rows)
            for symbol, rows in pair_rows.items()
            if rows
        }
        pair_walk_forward_scorecards = {
            symbol: build_walk_forward_scorecards(rows)
            for symbol, rows in pair_rows.items()
            if rows
        }
        lane_walk_forward_scorecards = {
            lane: build_walk_forward_scorecards(rows)
            for lane, rows in lane_rows.items()
            if rows
        }
        pair_loss_attribution = {
            symbol: build_loss_attribution_summary(rows)
            for symbol, rows in pair_rows.items()
            if rows
        }
        lane_loss_attribution = {
            lane: build_loss_attribution_summary(rows)
            for lane, rows in lane_rows.items()
            if rows
        }
        pair_lane_state = {
            symbol: resolve_lane_lifecycle(scorecards)
            for symbol, scorecards in pair_walk_forward_scorecards.items()
        }
        lane_state_matrix = {
            lane: resolve_lane_lifecycle(scorecards)
            for lane, scorecards in lane_walk_forward_scorecards.items()
        }
        weak_pairs = [
            str(symbol)
            for symbol, _stats in sorted(
                pair_stats.items(),
                key=lambda item: (
                    _safe_float((item[1] or {}).get("expectancy_r"), 0.0),
                    _safe_float((item[1] or {}).get("win_rate"), 0.0),
                    _safe_int((item[1] or {}).get("trade_count"), 0),
                    item[0],
                ),
            )
            if _safe_int((_stats or {}).get("trade_count"), 0) > 0
        ][:5]
        if not weak_pairs:
            weak_pairs = [
                str(symbol)
                for symbol, count in sorted(symbol_counts.items(), key=lambda item: (item[1], item[0]))
                if count > 0
            ][:5]
        weak_pair_lanes = [
            {
                "lane": lane_key,
                **self._pair_stats(rows),
            }
            for lane_key, rows in sorted(
                lane_rows.items(),
                key=lambda item: (
                    _safe_float(self._pair_stats(item[1]).get("expectancy_r"), 0.0),
                    _safe_float(self._pair_stats(item[1]).get("win_rate"), 0.0),
                    _safe_int(self._pair_stats(item[1]).get("trade_count"), 0),
                    item[0],
                ),
            )
        ][:8]
        setup_frame = local_memory.get("setup_frame")
        missed_opportunity_summary = self._summarize_missed_opportunities(setup_frame)
        news_regime_summary = self._summarize_news_regime(local_memory)
        proof_window = self._proof_window_summary(closed_trades)
        lane_hour_expectancy_matrix = self._build_lane_hour_expectancy_matrix(closed_trades)
        symbol_hour_expectancy_summary = self._summarize_symbol_hour_expectancy(lane_hour_expectancy_matrix)
        recent_lane_session_summary = self._recent_lane_session_summary(closed_trades)
        management_feedback_summary = self._summarize_management_feedback(live_feedback_rows)
        daily_rows = list(proof_window.get("daily_rows") or [])
        current_day_trade_count = int((daily_rows[-1] or {}).get("trade_count", 0) or 0) if daily_rows else 0
        daily_trade_target = 30
        undertrade_pressure = clamp(
            max(0.0, float(daily_trade_target - current_day_trade_count)) / max(float(daily_trade_target), 1.0),
            0.0,
            1.0,
        )
        latest_day_row = dict(daily_rows[-1] or {}) if daily_rows else {}
        profit_recycle_active = bool(
            latest_day_row.get("good_day", False)
            and current_day_trade_count >= 3
            and _safe_float(latest_day_row.get("expectancy_r"), 0.0) >= 0.0
            and win_rate >= 0.60
            and expectancy_r >= 0.0
        )
        profit_recycle_boost = clamp(
            (0.05 if profit_recycle_active else 0.0)
            + (0.05 if _safe_float(latest_day_row.get("expectancy_r"), 0.0) >= 0.35 else 0.0),
            0.0,
            0.15,
        )
        return {
            "reviewed_trade_count": int(len(closed_trades)),
            "win_rate": float(win_rate),
            "expectancy_r": float(expectancy_r),
            "profit_factor": float(profit_factor),
            "winner_loss_ratio": float(winner_loss_ratio),
            "drawdown_penalty": float(drawdown_penalty),
            "median_pnl_r": float(_median(pnl_r)),
            "mean_pnl_r": float(expectancy_r),
            "weak_pair_focus": weak_pairs,
            "pair_stats": pair_stats,
            "pair_walk_forward_scorecards": pair_walk_forward_scorecards,
            "lane_walk_forward_scorecards": lane_walk_forward_scorecards,
            "pair_loss_attribution": pair_loss_attribution,
            "lane_loss_attribution": lane_loss_attribution,
            "pair_lane_state": pair_lane_state,
            "lane_state_matrix": lane_state_matrix,
            "weak_pair_lanes": weak_pair_lanes,
            "winning_pattern_memory": [str(key) for key, _count in winning_setup_counter.most_common(12)],
            "proven_reentry_queue": [str(key) for key, _count in reentry_counter.most_common(12)],
            "failure_reasons": dict(failure_counter.most_common(8)),
            "missed_opportunity_summary": missed_opportunity_summary,
            "news_regime_summary": news_regime_summary,
            "new_pair_observation_summary": observation_summary,
            "proof_window": proof_window,
            "lane_hour_expectancy_matrix": lane_hour_expectancy_matrix,
            "symbol_hour_expectancy_summary": symbol_hour_expectancy_summary,
            "recent_lane_session_summary": recent_lane_session_summary,
            "management_feedback_summary": management_feedback_summary,
            "current_day_trade_count": int(current_day_trade_count),
            "daily_trade_target": int(daily_trade_target),
            "undertrade_fix_mode": bool(current_day_trade_count < daily_trade_target),
            "undertrade_pressure": float(undertrade_pressure),
            "profit_recycle_active": bool(profit_recycle_active),
            "profit_recycle_boost": float(profit_recycle_boost),
        }

    @staticmethod
    def _pair_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        pnl_values = [_safe_float(row.get("pnl_r"), 0.0) for row in rows]
        wins = [value for value in pnl_values if value >= 0.0]
        losses = [abs(value) for value in pnl_values if value < 0.0]
        return {
            "trade_count": int(len(pnl_values)),
            "win_rate": float((len(wins) / len(pnl_values)) if pnl_values else 0.0),
            "expectancy_r": float(_mean(pnl_values)),
            "profit_factor": float((sum(wins) / sum(losses)) if losses else (float("inf") if wins else 0.0)),
            "winner_loss_ratio": float((_mean(wins) / _mean(losses)) if wins and losses else (_mean(wins) if wins else 0.0)),
        }

    @staticmethod
    def _failure_reason(row: dict[str, Any]) -> str:
        spread_points = _safe_float(row.get("spread_points"), 0.0)
        news_state = str(row.get("news_state") or "").lower()
        regime_fit = _safe_float(row.get("regime_fit"), 1.0)
        entry_timing = _safe_float(row.get("entry_timing_score"), 1.0)
        structure = _safe_float(row.get("structure_cleanliness_score"), 1.0)
        continuation = _safe_float(row.get("swing_continuation_score"), 1.0)
        stop_distance = abs(_safe_float(row.get("entry"), 0.0) - _safe_float(row.get("sl"), 0.0))
        if "unknown" in news_state or "distortion" in news_state or "block" in news_state:
            return "news_distortion"
        if spread_points >= 25.0:
            return "spread_distortion"
        if regime_fit > 0.0 and regime_fit < 0.45:
            return "regime_mismatch"
        if continuation > 0.0 and continuation < 0.35:
            return "poor_continuation"
        if stop_distance > 0.0 and _safe_float(row.get("entry"), 0.0) > 0.0:
            if (stop_distance / max(abs(_safe_float(row.get("entry"), 0.0)), 1.0)) > 0.008:
                return "stop_too_wide"
        if (entry_timing > 0.0 and entry_timing < 0.45) or (structure > 0.0 and structure < 0.45):
            return "low_confluence"
        return "timing_or_structure"

    @staticmethod
    def _summarize_missed_opportunities(setup_frame: Any) -> dict[str, Any]:
        if not isinstance(setup_frame, pd.DataFrame) or setup_frame.empty:
            return {"total": 0, "top_lanes": []}
        frame = setup_frame.copy()
        decision_col = frame["decision"].astype(str).str.lower() if "decision" in frame.columns else pd.Series(dtype=str)
        accepted_col = frame["accepted"].astype(float) if "accepted" in frame.columns else pd.Series(dtype=float)
        mask = pd.Series([False] * len(frame))
        if not decision_col.empty:
            mask = decision_col.str.contains("reject|pass|block|hold", na=False)
        if not accepted_col.empty:
            mask = mask | (accepted_col <= 0.0)
        frame = frame.loc[mask]
        if frame.empty:
            return {"total": 0, "top_lanes": []}
        frame["lane"] = (
            frame.get("symbol", pd.Series([""] * len(frame))).astype(str).str.upper()
            + ":"
            + frame.get("strategy_key", frame.get("setup", pd.Series([""] * len(frame)))).astype(str).str.upper()
        )
        counts = frame["lane"].value_counts()
        return {
            "total": int(len(frame)),
            "top_lanes": [str(item) for item in counts.head(8).index.tolist()],
        }

    @staticmethod
    def _summarize_news_regime(local_memory: dict[str, Any]) -> dict[str, Any]:
        history_frame = local_memory.get("history_frame")
        if isinstance(history_frame, pd.DataFrame) and not history_frame.empty and "news_state" in history_frame.columns:
            counts = history_frame["news_state"].astype(str).str.lower().value_counts()
            return {str(key): int(value) for key, value in counts.head(8).items()}
        closed_trades = list(local_memory.get("closed_trades") or [])
        counts: Counter[str] = Counter(str(row.get("news_state") or "unknown").lower() for row in closed_trades)
        return {str(key): int(value) for key, value in counts.most_common(8)}

    @staticmethod
    def _proof_window_summary(closed_trades: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in closed_trades:
            closed_at_raw = (
                row.get("closed_at")
                or row.get("timestamp_utc")
                or row.get("opened_at")
                or row.get("created_at")
            )
            if not closed_at_raw:
                continue
            try:
                closed_at = pd.Timestamp(closed_at_raw)
                if closed_at.tzinfo is None:
                    closed_at = closed_at.tz_localize("UTC")
                closed_at = closed_at.tz_convert(SYDNEY)
            except Exception:
                continue
            grouped.setdefault(closed_at.date().isoformat(), []).append(dict(row))
        if not grouped:
            return {
                "days_observed": 0,
                "good_days_last_5": 0,
                "proof_ready_5d": False,
                "daily_rows": [],
            }
        daily_rows: list[dict[str, Any]] = []
        for day_key in sorted(grouped.keys())[-5:]:
            stats = ApexLearningBrain._pair_stats(grouped[day_key])
            good_day = bool(
                int(stats.get("trade_count", 0) or 0) > 0
                and float(stats.get("win_rate", 0.0) or 0.0) >= 0.60
                and float(stats.get("expectancy_r", 0.0) or 0.0) >= 0.0
                and float(stats.get("profit_factor", 0.0) or 0.0) >= 1.05
            )
            daily_rows.append(
                {
                    "day_key": day_key,
                    "trade_count": int(stats.get("trade_count", 0) or 0),
                    "win_rate": float(stats.get("win_rate", 0.0) or 0.0),
                    "expectancy_r": float(stats.get("expectancy_r", 0.0) or 0.0),
                    "profit_factor": float(stats.get("profit_factor", 0.0) or 0.0),
                    "good_day": bool(good_day),
                }
            )
        good_days_last_5 = sum(1 for row in daily_rows if bool(row.get("good_day")))
        return {
            "days_observed": int(len(grouped)),
            "good_days_last_5": int(good_days_last_5),
            "proof_ready_5d": bool(len(daily_rows) >= 5 and good_days_last_5 >= 5),
            "daily_rows": daily_rows,
        }

    @classmethod
    def _recent_lane_session_summary(cls, closed_trades: list[dict[str, Any]]) -> dict[str, Any]:
        fallback_closed_at = datetime.min.replace(tzinfo=SYDNEY)
        lane_session_groups: dict[str, dict[str, dict[str, Any]]] = {}
        for raw_row in closed_trades:
            row = dict(raw_row)
            symbol_key = _symbol_key(row.get("symbol"))
            setup_name = str(row.get("setup") or row.get("strategy_key") or row.get("symbol") or "").strip().upper()
            session_name = str(row.get("session_name") or "").strip().upper()
            closed_at = cls._row_closed_at_sydney(row)
            if not symbol_key or not setup_name or not session_name or closed_at is None:
                continue
            lane_key = f"{symbol_key}:{setup_name}"
            bucket_key = f"{closed_at.date().isoformat()}:{session_name}"
            bucket = lane_session_groups.setdefault(lane_key, {}).setdefault(
                bucket_key,
                {
                    "session_name": session_name,
                    "closed_at": closed_at,
                    "rows": [],
                },
            )
            bucket["rows"].append(row)
            if closed_at > bucket["closed_at"]:
                bucket["closed_at"] = closed_at

        lane_summary: dict[str, dict[str, Any]] = {}
        symbol_events: dict[str, list[dict[str, Any]]] = {}
        for lane_key, buckets in lane_session_groups.items():
            symbol_key, _, _setup_name = lane_key.partition(":")
            events: list[dict[str, Any]] = []
            for payload in buckets.values():
                rows = list(payload.get("rows") or [])
                stats = cls._pair_stats(rows)
                expectancy_r = float(stats.get("expectancy_r", 0.0) or 0.0)
                win_rate = float(stats.get("win_rate", 0.0) or 0.0)
                profit_factor = float(stats.get("profit_factor", 0.0) or 0.0)
                trade_count = int(stats.get("trade_count", 0) or 0)
                strong_session = bool(
                    trade_count > 0
                    and win_rate >= 0.60
                    and expectancy_r >= 0.0
                    and profit_factor >= 1.05
                )
                event = {
                    "session_name": str(payload.get("session_name") or ""),
                    "closed_at": payload.get("closed_at"),
                    "strong_session": bool(strong_session),
                    "trade_count": trade_count,
                    "win_rate": win_rate,
                    "expectancy_r": expectancy_r,
                }
                events.append(event)
                symbol_events.setdefault(symbol_key, []).append(
                    {
                        "lane_key": lane_key,
                        **event,
                    }
                )
            events.sort(key=lambda item: item.get("closed_at") or fallback_closed_at, reverse=True)
            consecutive_strong_sessions = 0
            for event in events:
                if bool(event.get("strong_session")):
                    consecutive_strong_sessions += 1
                    continue
                break
            latest = dict(events[0]) if events else {}
            hot_hand_score = clamp(
                (0.25 * min(consecutive_strong_sessions, 3))
                + (0.22 * max(0.0, float(latest.get("expectancy_r", 0.0) or 0.0)))
                + (0.28 * max(0.0, float(latest.get("win_rate", 0.0) or 0.0) - 0.50))
                + (0.08 if bool(latest.get("strong_session")) else 0.0),
                0.0,
                1.0,
            )
            lane_summary[lane_key] = {
                "recent_sessions_tracked": int(len(events)),
                "latest_session_name": str(latest.get("session_name") or ""),
                "latest_session_strong": bool(latest.get("strong_session", False)),
                "consecutive_strong_sessions": int(consecutive_strong_sessions),
                "hot_hand_active": bool(consecutive_strong_sessions >= 2),
                "hot_hand_score": float(hot_hand_score),
            }

        symbol_summary: dict[str, dict[str, Any]] = {}
        for symbol_key, events in symbol_events.items():
            ordered = sorted(events, key=lambda item: item.get("closed_at") or fallback_closed_at, reverse=True)
            max_consecutive = 0
            current_consecutive = 0
            session_scores: dict[str, list[float]] = {}
            for event in ordered:
                session_name = str(event.get("session_name") or "")
                session_score = clamp(
                    0.50
                    + (0.22 * max(0.0, float(event.get("expectancy_r", 0.0) or 0.0)))
                    + (0.24 * max(0.0, float(event.get("win_rate", 0.0) or 0.0) - 0.50))
                    + (0.10 if bool(event.get("strong_session", False)) else -0.04),
                    0.0,
                    1.0,
                )
                if session_name:
                    session_scores.setdefault(session_name, []).append(float(session_score))
                if bool(event.get("strong_session", False)):
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            session_score_summary = {
                str(session_name): float(clamp(_mean(scores), 0.0, 1.0))
                for session_name, scores in session_scores.items()
            }
            preferred_sessions = [
                str(name)
                for name, _score in sorted(
                    session_score_summary.items(),
                    key=lambda item: (-float(item[1]), item[0]),
                )
                if float(_score) >= 0.55
            ][:4]
            lane_hot_scores = [
                _safe_float(summary.get("hot_hand_score"), 0.0)
                for lane_key, summary in lane_summary.items()
                if lane_key.startswith(f"{symbol_key}:")
            ]
            hot_hand_score = clamp(
                max(lane_hot_scores or [0.0])
                + (0.06 if max_consecutive >= 2 else 0.0),
                0.0,
                1.0,
            )
            symbol_summary[symbol_key] = {
                "max_consecutive_strong_sessions": int(max_consecutive),
                "hot_hand_active": bool(max_consecutive >= 2),
                "hot_hand_score": float(hot_hand_score),
                "session_scores": session_score_summary,
                "preferred_sessions": preferred_sessions,
            }
        return {
            "lane_summary": lane_summary,
            "symbol_summary": symbol_summary,
        }

    @staticmethod
    def _row_closed_at_sydney(row: dict[str, Any]) -> datetime | None:
        closed_at_raw = (
            row.get("closed_at")
            or row.get("timestamp_utc")
            or row.get("opened_at")
            or row.get("created_at")
        )
        if not closed_at_raw:
            return None
        try:
            closed_at = pd.Timestamp(closed_at_raw)
            if closed_at.tzinfo is None:
                closed_at = closed_at.tz_localize("UTC")
            return closed_at.tz_convert(SYDNEY).to_pydatetime()
        except Exception:
            return None

    @staticmethod
    def _row_mfe_r(row: dict[str, Any]) -> float:
        explicit = row.get("mfe_r")
        if explicit is not None:
            return max(0.0, _safe_float(explicit, 0.0))
        pnl_r = _safe_float(row.get("pnl_r"), 0.0)
        return max(0.0, pnl_r)

    @staticmethod
    def _row_mae_r(row: dict[str, Any]) -> float:
        explicit = row.get("mae_r")
        if explicit is not None:
            return max(0.0, abs(_safe_float(explicit, 0.0)))
        pnl_r = _safe_float(row.get("pnl_r"), 0.0)
        if pnl_r < 0.0:
            return abs(pnl_r)
        return max(0.05, min(0.35, abs(pnl_r) * 0.35))

    @classmethod
    def _lane_exit_summary(cls, rows: list[dict[str, Any]]) -> dict[str, float]:
        mfe_values = [cls._row_mfe_r(row) for row in rows]
        mae_values = [cls._row_mae_r(row) for row in rows]
        captured = [
            clamp(_safe_float(row.get("pnl_r"), 0.0) / max(cls._row_mfe_r(row), 0.25), -1.0, 1.0)
            for row in rows
            if cls._row_mfe_r(row) > 0.0
        ]
        return {
            "lane_mfe_median_r": float(_median(mfe_values)),
            "lane_mae_median_r": float(_median(mae_values)),
            "lane_capture_efficiency": float(clamp(_mean(captured) if captured else 0.5, 0.0, 1.0)),
        }

    @classmethod
    def _build_lane_hour_expectancy_matrix(cls, closed_trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        lane_hours: dict[str, dict[int, list[dict[str, Any]]]] = {}
        lane_rows: dict[str, list[dict[str, Any]]] = {}
        for raw_row in closed_trades:
            row = dict(raw_row)
            symbol_key = _symbol_key(row.get("symbol"))
            setup_name = str(row.get("setup") or row.get("strategy_key") or row.get("symbol") or "").strip().upper()
            if not symbol_key or not setup_name:
                continue
            closed_at = cls._row_closed_at_sydney(row)
            if closed_at is None:
                continue
            lane_key = f"{symbol_key}:{setup_name}"
            lane_hours.setdefault(lane_key, {}).setdefault(int(closed_at.hour), []).append(row)
            lane_rows.setdefault(lane_key, []).append(row)
        matrix: dict[str, dict[str, Any]] = {}
        for lane_key, hour_map in sorted(lane_hours.items()):
            symbol_key, _, setup_name = lane_key.partition(":")
            all_rows = lane_rows.get(lane_key, [])
            lane_stats = cls._pair_stats(all_rows)
            exit_summary = cls._lane_exit_summary(all_rows)
            hour_rows: list[dict[str, Any]] = []
            strong_hours: list[int] = []
            weak_hours: list[int] = []
            for hour in sorted(hour_map.keys()):
                rows = hour_map[hour]
                hour_stats = cls._pair_stats(rows)
                hour_expectancy = float(hour_stats.get("expectancy_r", 0.0) or 0.0)
                hour_win_rate = float(hour_stats.get("win_rate", 0.0) or 0.0)
                trade_count = int(hour_stats.get("trade_count", 0) or 0)
                quality_tier = _hour_quality_tier(
                    expectancy_r=hour_expectancy,
                    win_rate=hour_win_rate,
                    trade_count=trade_count,
                )
                if quality_tier in {"A+", "A"}:
                    strong_hours.append(int(hour))
                elif quality_tier == "C":
                    weak_hours.append(int(hour))
                hour_rows.append(
                    {
                        "hour_sydney": int(hour),
                        "trade_count": trade_count,
                        "win_rate": hour_win_rate,
                        "expectancy_r": hour_expectancy,
                        "profit_factor": float(hour_stats.get("profit_factor", 0.0) or 0.0),
                        "quality_tier": quality_tier,
                    }
                )
            hour_expectancy_score = clamp(
                0.50
                + (float(lane_stats.get("expectancy_r", 0.0) or 0.0) * 0.22)
                + ((float(lane_stats.get("win_rate", 0.0) or 0.0) - 0.50) * 0.55)
                + ((float(exit_summary.get("lane_capture_efficiency", 0.5) or 0.5) - 0.50) * 0.20),
                0.0,
                1.0,
            )
            lane_expectancy_multiplier = clamp(
                0.88
                + ((hour_expectancy_score - 0.50) * 0.50)
                + (0.03 if strong_hours else 0.0)
                - (0.04 if weak_hours and not strong_hours else 0.0),
                0.80,
                1.35,
            )
            matrix[lane_key] = {
                "symbol": symbol_key,
                "setup": setup_name,
                "hour_rows": hour_rows,
                "strong_hours_sydney": strong_hours[:6],
                "weak_hours_sydney": weak_hours[:6],
                "hour_expectancy_score": float(hour_expectancy_score),
                "lane_expectancy_multiplier": float(lane_expectancy_multiplier),
                "lane_mfe_median_r": float(exit_summary.get("lane_mfe_median_r", 0.0) or 0.0),
                "lane_mae_median_r": float(exit_summary.get("lane_mae_median_r", 0.0) or 0.0),
                "lane_capture_efficiency": float(exit_summary.get("lane_capture_efficiency", 0.5) or 0.5),
                "trade_count": int(lane_stats.get("trade_count", 0) or 0),
                "win_rate": float(lane_stats.get("win_rate", 0.0) or 0.0),
                "expectancy_r": float(lane_stats.get("expectancy_r", 0.0) or 0.0),
            }
        return matrix

    @staticmethod
    def _summarize_symbol_hour_expectancy(lane_matrix: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        symbol_map: dict[str, list[dict[str, Any]]] = {}
        for lane_summary in lane_matrix.values():
            symbol_key = str(lane_summary.get("symbol") or "").upper()
            if not symbol_key:
                continue
            symbol_map.setdefault(symbol_key, []).append(dict(lane_summary))
        summaries: dict[str, dict[str, Any]] = {}
        for symbol_key, rows in symbol_map.items():
            strong_counter: Counter[int] = Counter()
            weak_counter: Counter[int] = Counter()
            for row in rows:
                strong_counter.update(int(item) for item in row.get("strong_hours_sydney", []) or [])
                weak_counter.update(int(item) for item in row.get("weak_hours_sydney", []) or [])
            summaries[symbol_key] = {
                "hour_expectancy_score": float(clamp(_mean([_safe_float(row.get("hour_expectancy_score"), 0.5) for row in rows]), 0.0, 1.0)),
                "lane_expectancy_multiplier": float(
                    clamp(_mean([_safe_float(row.get("lane_expectancy_multiplier"), 1.0) for row in rows]), 0.80, 1.35)
                ),
                "lane_mfe_median_r": float(_median([_safe_float(row.get("lane_mfe_median_r"), 0.0) for row in rows])),
                "lane_mae_median_r": float(_median([_safe_float(row.get("lane_mae_median_r"), 0.0) for row in rows])),
                "lane_capture_efficiency": float(
                    clamp(_mean([_safe_float(row.get("lane_capture_efficiency"), 0.5) for row in rows]), 0.0, 1.0)
                ),
                "strong_hours_sydney": [int(hour) for hour, _count in strong_counter.most_common(6)],
                "weak_hours_sydney": [int(hour) for hour, _count in weak_counter.most_common(6)],
                "tracked_lanes": int(len(rows)),
            }
        return summaries

    @staticmethod
    def _summarize_management_feedback(rows: list[dict[str, Any]]) -> dict[str, Any]:
        symbol_groups: dict[str, list[dict[str, Any]]] = {}
        lane_groups: dict[str, list[dict[str, Any]]] = {}
        for raw_row in rows:
            row = dict(raw_row)
            symbol_key = _symbol_key(row.get("symbol"))
            setup_name = str(row.get("setup") or row.get("strategy_key") or "").strip().upper()
            if not symbol_key:
                continue
            symbol_groups.setdefault(symbol_key, []).append(row)
            if setup_name:
                lane_groups.setdefault(f"{symbol_key}:{setup_name}", []).append(row)

        def _group_summary(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
            profitable_rows = [row for row in group_rows if _safe_float(row.get("pnl_r"), -9.0) >= 0.0]
            actionable_rows = [
                row for row in profitable_rows if str(row.get("management_action") or "").upper() not in {"", "HOLD"}
            ]
            trail_rows = [row for row in actionable_rows if str(row.get("management_action") or "").upper() == "TRAIL_STOP"]
            tp_rows = [row for row in actionable_rows if str(row.get("management_action") or "").upper() == "EXTEND_TP"]
            early_rows = [row for row in actionable_rows if "early_protect" in str(row.get("management_reason") or "").lower()]
            gaps = [
                max(0.0, _safe_float(row.get("mfe_r"), 0.0) - _safe_float(row.get("pnl_r"), 0.0))
                for row in profitable_rows
            ]
            capture_samples = [
                clamp(_safe_float(row.get("pnl_r"), 0.0) / max(_safe_float(row.get("mfe_r"), 0.0), 0.25), 0.0, 1.25)
                for row in profitable_rows
                if _safe_float(row.get("mfe_r"), 0.0) > 0.0
            ]
            trail_giveback_samples = [
                max(0.0, _safe_float(row.get("mfe_r"), 0.0) - _safe_float(row.get("pnl_r"), 0.0))
                for row in trail_rows
            ]
            extension_successes = [
                row
                for row in tp_rows
                if _safe_float(row.get("pnl_r"), 0.0)
                >= max(0.90, _safe_float(row.get("mfe_r"), 0.0) * 0.70)
            ]
            cadence_samples: list[float] = []
            action_times = []
            for row in actionable_rows:
                raw = row.get("logged_at")
                if not raw:
                    continue
                try:
                    ts = pd.Timestamp(raw)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    action_times.append(ts)
                except Exception:
                    continue
            action_times = sorted(action_times)
            for previous, current in zip(action_times, action_times[1:]):
                cadence_samples.append(max(0.0, float((current - previous).total_seconds())))
            close_winners_score = clamp(
                0.50
                + ((_mean(capture_samples) - 0.50) * 0.55 if capture_samples else 0.0)
                + ((len(extension_successes) / max(len(tp_rows), 1)) * 0.16 if tp_rows else 0.0)
                - (min(0.18, _median(trail_giveback_samples) * 0.12) if trail_giveback_samples else 0.0),
                0.0,
                1.0,
            )
            return {
                "feedback_rows": int(len(group_rows)),
                "profitable_feedback_rows": int(len(profitable_rows)),
                "actionable_profit_rows": int(len(actionable_rows)),
                "active_management_ratio": float(
                    clamp((len(actionable_rows) / max(len(profitable_rows), 1)), 0.0, 1.0)
                ),
                "trail_update_rate": float(clamp((len(trail_rows) / max(len(profitable_rows), 1)), 0.0, 1.0)),
                "tp_extension_rate": float(clamp((len(tp_rows) / max(len(profitable_rows), 1)), 0.0, 1.0)),
                "tp_extension_success_rate": float(clamp((len(extension_successes) / max(len(tp_rows), 1)), 0.0, 1.0)),
                "early_protect_rate": float(clamp((len(early_rows) / max(len(profitable_rows), 1)), 0.0, 1.0)),
                "missed_runner_gap_r": float(_median(gaps)),
                "mean_action_cadence_seconds": float(_median(cadence_samples)),
                "avg_mfe_capture_rate": float(clamp(_mean(capture_samples) if capture_samples else 0.5, 0.0, 1.0)),
                "trail_giveback_r": float(_median(trail_giveback_samples)),
                "close_winners_score": float(close_winners_score),
            }

        symbol_summary = {
            symbol_key: _group_summary(group_rows)
            for symbol_key, group_rows in sorted(symbol_groups.items())
        }
        lane_summary = {
            lane_key: _group_summary(group_rows)
            for lane_key, group_rows in sorted(lane_groups.items())
        }
        undermanaged_symbols = [
            symbol_key
            for symbol_key, summary in symbol_summary.items()
            if float(summary.get("profitable_feedback_rows", 0) or 0) >= 3
            and float(summary.get("active_management_ratio", 0.0) or 0.0) < 0.35
        ]
        return {
            "symbol_summary": symbol_summary,
            "lane_summary": lane_summary,
            "undermanaged_symbols": undermanaged_symbols,
        }

    @staticmethod
    def _pair_native_sessions(symbol: str) -> list[str]:
        symbol_key = _symbol_key(symbol)
        if symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD", "USDJPY", "EURJPY", "GBPJPY"}:
            return ["SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"]
        if symbol_key in {"EURUSD", "GBPUSD", "EURGBP", "NAS100", "USOIL"}:
            return ["LONDON", "OVERLAP", "NEW_YORK"]
        if symbol_key == "XAUUSD":
            return ["SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"]
        if symbol_key == "BTCUSD":
            return ["SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"]
        return ["LONDON", "OVERLAP", "NEW_YORK"]

    @staticmethod
    def _asia_native_pairs() -> set[str]:
        return {"AUDJPY", "NZDJPY", "AUDNZD", "USDJPY", "EURJPY", "GBPJPY"}

    @staticmethod
    def _normalize_hybrid_pair_ideas(payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, list):
            return []
        ideas: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            symbol_key = _symbol_key(item.get("symbol"))
            if not symbol_key:
                continue
            session_focus = [
                str(value).upper()
                for value in item.get("session_focus", [])
                if str(value).strip()
            ]
            ideas.append(
                {
                    "symbol": symbol_key,
                    "session_focus": list(dict.fromkeys(session_focus)),
                    "setup_bias": str(item.get("setup_bias") or "").strip().upper(),
                    "direction_bias": str(item.get("direction_bias") or "").strip().upper(),
                    "conviction": float(clamp(_safe_float(item.get("conviction"), 0.0), 0.0, 1.0)),
                    "aggression_delta": float(clamp(_safe_float(item.get("aggression_delta"), 0.0), -0.10, 0.25)),
                    "threshold_delta": float(clamp(_safe_float(item.get("threshold_delta"), 0.0), -0.08, 0.05)),
                    "reason": str(item.get("reason") or "").strip(),
                }
            )
        return ideas[:12]

    def _apply_gpt_hybrid_pair_ideas(
        self,
        *,
        pair_directives: dict[str, dict[str, Any]],
        offline_review: OfflineGPTReview,
        current_session_name: str,
    ) -> dict[str, dict[str, Any]]:
        directives: dict[str, dict[str, Any]] = {
            str(symbol): {
                **dict(directive or {}),
                "management_directives": dict((directive or {}).get("management_directives") or {}),
                "frequency_directives": dict((directive or {}).get("frequency_directives") or {}),
                "blocker_relaxation_state": dict((directive or {}).get("blocker_relaxation_state") or {}),
                "targeted_backtest_probe": dict((directive or {}).get("targeted_backtest_probe") or {}),
            }
            for symbol, directive in (pair_directives or {}).items()
        }
        current_session_key = str(current_session_name or "").upper()
        for idea in list(offline_review.hybrid_pair_ideas or []):
            symbol_key = _symbol_key(idea.get("symbol"))
            directive = directives.get(symbol_key)
            if not directive:
                continue
            session_focus = list(directive.get("session_focus") or self._pair_native_sessions(symbol_key))
            hybrid_sessions = [str(value).upper() for value in idea.get("session_focus", []) if str(value).strip()]
            if hybrid_sessions:
                session_focus = list(dict.fromkeys(session_focus + hybrid_sessions))
            conviction = clamp(_safe_float(idea.get("conviction"), 0.0), 0.0, 1.0)
            aggression_delta = clamp(_safe_float(idea.get("aggression_delta"), 0.0), -0.10, 0.25)
            threshold_delta = clamp(_safe_float(idea.get("threshold_delta"), 0.0), -0.08, 0.05)
            session_match = not hybrid_sessions or current_session_key in set(hybrid_sessions)
            directive["session_focus"] = session_focus
            directive["gpt_hybrid_advisory"] = {
                "enabled": bool(conviction >= 0.35),
                "conviction": float(conviction),
                "direction_bias": str(idea.get("direction_bias") or "").upper(),
                "setup_bias": str(idea.get("setup_bias") or "").upper(),
                "threshold_delta": float(threshold_delta),
                "aggression_delta": float(aggression_delta),
                "session_focus": list(hybrid_sessions or session_focus),
                "session_match": bool(session_match),
                "reason": str(idea.get("reason") or ""),
            }
            directive["aggression_multiplier"] = float(
                clamp(
                    _safe_float(directive.get("aggression_multiplier"), 1.0)
                    + (aggression_delta * (0.55 + (conviction * 0.45))),
                    0.75,
                    1.60,
                )
            )
            directive["min_confluence_override"] = float(
                clamp(
                    _safe_float(directive.get("min_confluence_override"), 3.4) + (threshold_delta * 2.0),
                    2.90,
                    4.40,
                )
            )
            frequency_directives = dict(directive.get("frequency_directives") or {})
            frequency_directives["catchup_pressure"] = float(
                clamp(
                    _safe_float(frequency_directives.get("catchup_pressure"), 0.0)
                    + max(0.0, aggression_delta * 0.40)
                    + (conviction * 0.08),
                    0.0,
                    0.95,
                )
            )
            if session_match and conviction >= 0.55:
                frequency_directives["quota_boost_allowed"] = True
                frequency_directives["aggressive_reentry_enabled"] = True
                if symbol_key in self._asia_native_pairs() and current_session_key in {"SYDNEY", "TOKYO"}:
                    frequency_directives["soft_burst_target_10m"] = max(
                        int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
                        5,
                    )
                elif symbol_key == "BTCUSD":
                    frequency_directives["soft_burst_target_10m"] = max(
                        int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
                        7,
                    )
            directive["frequency_directives"] = frequency_directives
            directives[symbol_key] = directive
        return directives

    def _build_pair_directives(
        self,
        *,
        local_summary: dict[str, Any],
        runtime_state: dict[str, Any],
        learning_status: dict[str, Any],
        goal_state: dict[str, Any],
        current_session_name: str = "",
        shadow_strategy_variants: list[dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        pair_stats = dict(local_summary.get("pair_stats") or {})
        symbol_hour_expectancy_summary = dict(local_summary.get("symbol_hour_expectancy_summary") or {})
        recent_lane_session_summary = dict(local_summary.get("recent_lane_session_summary") or {})
        recent_symbol_session_summary = dict(recent_lane_session_summary.get("symbol_summary") or {})
        management_feedback_summary = dict(local_summary.get("management_feedback_summary") or {})
        management_symbol_summary = dict(management_feedback_summary.get("symbol_summary") or {})
        pair_walk_forward_scorecards = dict(local_summary.get("pair_walk_forward_scorecards") or {})
        pair_loss_attribution = dict(local_summary.get("pair_loss_attribution") or {})
        pair_lane_state = dict(local_summary.get("pair_lane_state") or {})
        symbol_runtime = dict(runtime_state.get("symbol_runtime") or {})
        trajectory_runtime = dict(runtime_state.get("xau_btc_trajectory_stats") or {})
        symbols = (
            set(pair_stats.keys())
            | {str(key).upper() for key in symbol_runtime.keys() if str(key).strip()}
            | {str(key).upper() for key in runtime_state.get("symbols", []) if str(key).strip()}
        )
        directives: dict[str, dict[str, Any]] = {}
        weak_pairs = {str(item).upper() for item in local_summary.get("weak_pair_focus", []) if str(item).strip()}
        missed_lanes = set(local_summary.get("missed_opportunity_summary", {}).get("top_lanes", []) or [])
        history_seed_ready = str(learning_status.get("last_market_history_seed_status") or "").lower() in {"ok", "not_needed"}
        current_session_key = str(current_session_name or "").upper()
        profit_recycle_active = bool(local_summary.get("profit_recycle_active", False))
        profit_recycle_boost = clamp(_safe_float(local_summary.get("profit_recycle_boost"), 0.0), 0.0, 0.25)
        for symbol_key in sorted(symbols):
            stats = dict(pair_stats.get(symbol_key) or {})
            hour_summary = dict(symbol_hour_expectancy_summary.get(symbol_key) or {})
            hot_hand_summary = dict(recent_symbol_session_summary.get(symbol_key) or {})
            runtime = dict(symbol_runtime.get(symbol_key) or {})
            trajectory_stats = dict(trajectory_runtime.get(symbol_key) or {})
            management_feedback = dict(management_symbol_summary.get(symbol_key) or {})
            walk_forward_scorecards = dict(pair_walk_forward_scorecards.get(symbol_key) or {})
            loss_attribution_summary = dict(pair_loss_attribution.get(symbol_key) or {})
            lane_state_machine = dict(pair_lane_state.get(symbol_key) or {})
            if not lane_state_machine and walk_forward_scorecards:
                lane_state_machine = resolve_lane_lifecycle(walk_forward_scorecards)
            shadow_challenger_pool = build_shadow_challenger_pool(
                symbol_key=symbol_key,
                shadow_strategy_variants=list(shadow_strategy_variants or []),
                lifecycle_state=lane_state_machine,
                current_session_name=current_session_name,
                limit=5,
            )
            news_direction = str(runtime.get("news_bias_direction") or runtime.get("bias_direction") or "neutral").lower()
            news_confidence = clamp(_safe_float(runtime.get("news_confidence"), 0.0), 0.0, 1.0)
            session_focus = self._pair_native_sessions(symbol_key)
            today_closed_trade_count = max(0, _safe_int(runtime.get("today_closed_trade_count"), 0))
            session_trade_count = max(0, _safe_int(runtime.get("session_trade_count"), 0))
            win_rate = clamp(_safe_float(stats.get("win_rate"), 0.0), 0.0, 1.0)
            expectancy_r = _safe_float(stats.get("expectancy_r"), 0.0)
            trade_count = _safe_int(stats.get("trade_count"), 0)
            weak_focus = symbol_key in weak_pairs
            lane_state_value = str(lane_state_machine.get("state") or "probation").lower()
            lane_shadow_only = bool(lane_state_machine.get("shadow_only", False))
            lane_ramp_multiplier = clamp(_safe_float(lane_state_machine.get("ramp_multiplier"), 1.0), 0.40, 1.30)
            lane_recovery_ready = bool(lane_state_machine.get("recovery_ready", False))
            lane_recent_edge_broken = bool(lane_state_machine.get("recent_edge_broken", False))
            lane_short_edge_broken = bool(lane_state_machine.get("short_edge_broken", False))
            slippage_quality_score = clamp(_safe_float(runtime.get("slippage_quality_score"), 0.55), 0.0, 1.0)
            broker_reject_streak = max(0, _safe_int(runtime.get("broker_reject_streak"), 0))
            opportunity_capture_gap = max(0.0, _safe_float(runtime.get("recent_opportunity_capture_gap_r"), 0.0))
            management_quality_score = clamp(_safe_float(runtime.get("recent_management_quality_score"), 0.0), 0.0, 1.0)
            if not runtime.get("recent_management_quality_score"):
                management_quality_score = clamp(_safe_float(management_feedback.get("active_management_ratio"), 0.0), 0.0, 1.0)
            opportunity_capture_gap = max(opportunity_capture_gap, _safe_float(management_feedback.get("missed_runner_gap_r"), 0.0))
            close_winners_score = clamp(_safe_float(management_feedback.get("close_winners_score"), management_quality_score), 0.0, 1.0)
            extension_success_rate = clamp(_safe_float(management_feedback.get("tp_extension_success_rate"), 0.0), 0.0, 1.0)
            trail_giveback_r = max(0.0, _safe_float(management_feedback.get("trail_giveback_r"), 0.0))
            recent_window = dict((walk_forward_scorecards.get("windows") or {}).get("1d") or {})
            recent_spread_points = _safe_float(recent_window.get("spread_p80_points"), _safe_float(recent_window.get("spread_avg_points"), 0.0))
            recent_slippage_points = _safe_float(recent_window.get("slippage_p80_points"), _safe_float(recent_window.get("slippage_avg_points"), 0.0))
            execution_quality_gate = evaluate_execution_quality_gate(
                spread_points=recent_spread_points,
                typical_spread_points=max(1.0, _safe_float(runtime.get("typical_spread_points"), recent_spread_points or 1.0)),
                stop_distance_points=max(8.0, _safe_float(runtime.get("avg_stop_distance_points"), 25.0)),
                slippage_quality_score=slippage_quality_score,
                execution_quality_score=clamp(_safe_float(runtime.get("execution_quality_score"), 0.65), 0.0, 1.0),
                microstructure_alignment=clamp(
                    _safe_float(runtime.get("microstructure_alignment_score"), _safe_float(runtime.get("microstructure_score"), 0.50)),
                    0.0,
                    1.0,
                ),
                adverse_entry_risk=clamp(_safe_float(runtime.get("adverse_entry_risk"), 0.25), 0.0, 1.0),
                lifecycle_state=lane_state_value,
            )
            shadow_experiment_active = bool(runtime.get("shadow_experiment_active")) or (
                bool(goal_state.get("lagging_short_goal")) and trade_count >= 3 and expectancy_r >= 0.0
            )
            hour_expectancy_score = clamp(_safe_float(hour_summary.get("hour_expectancy_score"), 0.5), 0.0, 1.0)
            lane_expectancy_multiplier = clamp(_safe_float(hour_summary.get("lane_expectancy_multiplier"), 1.0), 0.80, 1.35)
            strong_hours_sydney = [int(item) for item in hour_summary.get("strong_hours_sydney", []) if _safe_int(item, -1) >= 0][:6]
            weak_hours_sydney = [int(item) for item in hour_summary.get("weak_hours_sydney", []) if _safe_int(item, -1) >= 0][:6]
            session_scores = {
                str(key).upper(): clamp(_safe_float(value, 0.0), 0.0, 1.0)
                for key, value in dict(hot_hand_summary.get("session_scores") or {}).items()
                if str(key).strip()
            }
            preferred_sessions = [str(item).upper() for item in hot_hand_summary.get("preferred_sessions", []) if str(item).strip()]
            hot_hand_session_streak = max(0, _safe_int(hot_hand_summary.get("max_consecutive_strong_sessions"), 0))
            hot_hand_score = clamp(_safe_float(hot_hand_summary.get("hot_hand_score"), 0.0), 0.0, 1.0)
            session_heat_score = clamp(_safe_float(session_scores.get(current_session_key), 0.0), 0.0, 1.0)
            hot_hand_active = bool(
                hot_hand_session_streak >= 2
                and (
                    current_session_key in preferred_sessions
                    or session_heat_score >= 0.58
                    or not current_session_key
                )
            )
            lane_mfe_median_r = max(0.0, _safe_float(hour_summary.get("lane_mfe_median_r"), 0.0))
            lane_mae_median_r = max(0.0, _safe_float(hour_summary.get("lane_mae_median_r"), 0.0))
            lane_capture_efficiency = clamp(_safe_float(hour_summary.get("lane_capture_efficiency"), 0.5), 0.0, 1.0)
            runner_lane = bool(
                lane_mfe_median_r >= 1.75
                and lane_capture_efficiency >= 0.58
                and lane_expectancy_multiplier >= 1.02
            )
            stall_lane = bool(
                (lane_mfe_median_r > 0.0 and lane_mfe_median_r <= 1.05)
                or (lane_capture_efficiency <= 0.48 and lane_mfe_median_r > 0.0)
                or (hour_expectancy_score <= 0.42 and lane_mfe_median_r > 0.0)
            )
            proof_window = dict(local_summary.get("proof_window") or {})
            proof_ready_5d = bool(proof_window.get("proof_ready_5d", False))
            good_days_last_5 = max(0, _safe_int(proof_window.get("good_days_last_5"), 0))
            proof_lane_ready = bool(
                proof_ready_5d
                and trade_count >= 3
                and win_rate >= 0.60
                and expectancy_r >= 0.0
            )
            asia_native_pair = symbol_key in self._asia_native_pairs()
            xau_asia_session = bool(symbol_key == "XAUUSD" and current_session_key in {"SYDNEY", "TOKYO"})
            xau_prime_session = bool(symbol_key == "XAUUSD" and current_session_key in {"LONDON", "OVERLAP", "NEW_YORK"})
            lane_idle_today = bool(today_closed_trade_count <= 0 and session_trade_count <= 0)
            aggression_multiplier = 1.0
            if trade_count >= 3 and win_rate >= 0.60 and expectancy_r >= 0.0:
                aggression_multiplier += 0.12
            if bool(goal_state.get("lagging_short_goal")):
                aggression_multiplier += 0.10
            if weak_focus:
                aggression_multiplier -= 0.12
            if news_confidence >= 0.70 and news_direction in {"bullish", "bearish", "risk_on", "risk_off"}:
                aggression_multiplier += 0.06
            if symbol_key in {"USOIL", "NAS100"} and news_confidence >= 0.65 and news_direction in {"bullish", "bearish", "risk_on", "risk_off"}:
                aggression_multiplier += 0.10
            if symbol_key in {"USOIL", "NAS100"} and news_confidence >= 0.80 and expectancy_r >= 0.0:
                aggression_multiplier += 0.06
            if slippage_quality_score >= 0.72:
                aggression_multiplier += 0.05
            elif slippage_quality_score <= 0.38:
                aggression_multiplier -= 0.10
            if broker_reject_streak >= 2:
                aggression_multiplier -= 0.08
            if opportunity_capture_gap >= 0.60 and management_quality_score <= 0.55:
                aggression_multiplier += 0.03
            if hour_expectancy_score >= 0.62:
                aggression_multiplier += 0.06
            elif hour_expectancy_score <= 0.42:
                aggression_multiplier -= 0.08
            if lane_expectancy_multiplier >= 1.08:
                aggression_multiplier += 0.04
            elif lane_expectancy_multiplier <= 0.94:
                aggression_multiplier -= 0.05
            lagging_soft_target = bool(trajectory_stats.get("lagging", False))
            trajectory_catchup = clamp(_safe_float(trajectory_stats.get("catchup_pressure"), 0.0), 0.0, 1.0)
            if lagging_soft_target and symbol_key in {"XAUUSD", "BTCUSD"}:
                aggression_multiplier += 0.08
            if _safe_float(local_summary.get("undertrade_pressure"), 0.0) >= 0.35 and symbol_key in {"XAUUSD", "BTCUSD"}:
                aggression_multiplier += 0.06
            if (
                asia_native_pair
                and current_session_key in {"SYDNEY", "TOKYO"}
                and _safe_float(local_summary.get("undertrade_pressure"), 0.0) >= 0.20
            ):
                aggression_multiplier += 0.14
            if (
                lane_idle_today
                and current_session_key
                and (
                    current_session_key in session_focus
                    or (asia_native_pair and current_session_key in {"SYDNEY", "TOKYO"})
                    or symbol_key in {"XAUUSD", "BTCUSD"}
                )
                and _safe_float(local_summary.get("undertrade_pressure"), 0.0) >= 0.12
            ):
                aggression_multiplier += 0.08 if symbol_key in {"XAUUSD", "BTCUSD"} else 0.10
            if asia_native_pair and current_session_key in {"SYDNEY", "TOKYO"} and hour_expectancy_score >= 0.52:
                aggression_multiplier += 0.06
            if hot_hand_active:
                aggression_multiplier += 0.08 + min(0.05, hot_hand_score * 0.08)
            if session_heat_score >= 0.62:
                aggression_multiplier += 0.04
            if proof_lane_ready and not xau_asia_session:
                aggression_multiplier += 0.08
                if symbol_key in {"XAUUSD", "BTCUSD"}:
                    aggression_multiplier += 0.04
                elif asia_native_pair and current_session_key in {"SYDNEY", "TOKYO"}:
                    aggression_multiplier += 0.03
            if symbol_key == "XAUUSD" and xau_prime_session and (proof_lane_ready or hot_hand_active):
                aggression_multiplier += 0.08
            if symbol_key == "XAUUSD" and xau_prime_session and (weak_focus or expectancy_r < 0.0 or win_rate < 0.55):
                aggression_multiplier -= 0.06
            if profit_recycle_active and expectancy_r >= 0.0 and not xau_asia_session:
                aggression_multiplier += min(0.05, profit_recycle_boost * 0.40)
            if lane_state_value == "attack":
                aggression_multiplier += 0.10
            elif lane_state_value == "proven":
                aggression_multiplier += 0.05
            elif lane_state_value == "degrade":
                aggression_multiplier -= 0.22
            elif lane_state_value == "shadow_only":
                aggression_multiplier -= 0.42
            if lane_recent_edge_broken:
                aggression_multiplier -= 0.10
            if lane_short_edge_broken:
                aggression_multiplier -= 0.08
            if lane_state_value == "probation" and not lane_recovery_ready and trade_count >= 3:
                aggression_multiplier -= 0.05
            aggression_multiplier *= lane_ramp_multiplier
            if xau_asia_session:
                aggression_multiplier = min(aggression_multiplier, 1.0)
            aggression_multiplier = clamp(aggression_multiplier, 0.55, 1.45)
            if news_confidence >= 0.75 and news_direction in {"bullish", "bearish"} and expectancy_r >= 0.0:
                trade_horizon_bias = "swing"
            elif symbol_key not in {"XAUUSD", "BTCUSD"} and lane_mfe_median_r >= 1.35 and hour_expectancy_score >= 0.54:
                trade_horizon_bias = "daytrade"
            elif symbol_key in {"XAUUSD", "BTCUSD"} or win_rate >= 0.62:
                trade_horizon_bias = "scalp"
            else:
                trade_horizon_bias = "daytrade"
            if symbol_key in {"USOIL", "NAS100"} and news_confidence >= 0.65:
                trade_horizon_bias = "daytrade"
            if opportunity_capture_gap >= 0.70 and management_quality_score <= 0.50 and trade_horizon_bias == "scalp":
                trade_horizon_bias = "daytrade"
            min_confluence_override = 4.0 if weak_focus else 3.2 if aggression_multiplier >= 1.15 else 3.4
            if hour_expectancy_score <= 0.42:
                min_confluence_override += 0.20
            elif hour_expectancy_score >= 0.68 and strong_hours_sydney:
                min_confluence_override = max(3.0, min_confluence_override - 0.10)
            if (
                lane_idle_today
                and current_session_key
                and (
                    current_session_key in session_focus
                    or (asia_native_pair and current_session_key in {"SYDNEY", "TOKYO"})
                    or symbol_key in {"XAUUSD", "BTCUSD"}
                )
                and not xau_asia_session
            ):
                min_confluence_override = max(3.0, min_confluence_override - 0.12)
            if symbol_key == "XAUUSD" and xau_prime_session and (weak_focus or expectancy_r < 0.0 or win_rate < 0.55):
                min_confluence_override = max(min_confluence_override, 4.15)
            elif symbol_key == "XAUUSD" and xau_prime_session and (proof_lane_ready or hot_hand_active):
                min_confluence_override = max(3.0, min_confluence_override - 0.08)
            if lane_state_value == "degrade":
                min_confluence_override += 0.25
            elif lane_shadow_only:
                min_confluence_override = max(min_confluence_override, 4.35)
            if lane_recent_edge_broken:
                min_confluence_override += 0.15
            if lane_short_edge_broken:
                min_confluence_override += 0.10
            reentry_priority = clamp(
                0.40
                + (0.20 if trade_count >= 3 and win_rate >= 0.60 else 0.0)
                + (0.15 if any(item.startswith(f"{symbol_key}:") for item in missed_lanes) else 0.0),
                0.25,
                0.95,
            )
            early_protect_r = 0.24
            trail_cadence_seconds = 45
            trail_backoff_r = 0.34
            tp_extension_bias = 0.16
            stall_exit_bias = 0.28
            if symbol_key == "XAUUSD":
                early_protect_r = 0.12 if xau_prime_session else 0.18 if xau_asia_session else 0.14
                trail_cadence_seconds = 10 if xau_prime_session else 30 if xau_asia_session else 18
                trail_backoff_r = 0.20 if xau_prime_session else 0.28 if xau_asia_session else 0.24
                tp_extension_bias = 0.30 if xau_prime_session else 0.16 if xau_asia_session else 0.22
            elif symbol_key == "BTCUSD":
                early_protect_r = 0.16
                trail_cadence_seconds = 15
                trail_backoff_r = 0.36
                tp_extension_bias = 0.30
            elif symbol_key in {"NAS100", "USOIL"}:
                early_protect_r = 0.21
                trail_cadence_seconds = 35
                trail_backoff_r = 0.30
                tp_extension_bias = 0.22
            if runner_lane:
                early_protect_r += 0.02
                trail_backoff_r += 0.08
                tp_extension_bias += 0.10
                stall_exit_bias -= 0.10
            elif stall_lane:
                early_protect_r -= 0.06
                trail_backoff_r -= 0.06
                tp_extension_bias -= 0.06
                stall_exit_bias += 0.18
            if hour_expectancy_score >= 0.68 and strong_hours_sydney:
                early_protect_r += 0.02
                tp_extension_bias += 0.06
            elif hour_expectancy_score <= 0.42:
                early_protect_r -= 0.04
                stall_exit_bias += 0.10
            if opportunity_capture_gap >= 0.60 and management_quality_score <= 0.55:
                trail_backoff_r += 0.04
                tp_extension_bias += 0.05
            if _safe_float(management_feedback.get("active_management_ratio"), 0.0) < 0.35 and symbol_key in {"XAUUSD", "BTCUSD"}:
                early_protect_r -= 0.02
                trail_cadence_seconds -= 5
                tp_extension_bias += 0.05
            if _safe_float(management_feedback.get("trail_update_rate"), 0.0) < 0.15 and symbol_key == "XAUUSD":
                trail_cadence_seconds -= 5
                trail_backoff_r += 0.04
            if _safe_float(management_feedback.get("tp_extension_rate"), 0.0) < 0.10 and runner_lane:
                tp_extension_bias += 0.04
            if runner_lane and close_winners_score <= 0.48:
                trail_backoff_r += 0.03
                tp_extension_bias += 0.04
            if symbol_key in {"XAUUSD", "BTCUSD"} and runner_lane and opportunity_capture_gap >= 0.40:
                trail_backoff_r += 0.06
                tp_extension_bias += 0.07
                stall_exit_bias -= 0.06
            if runner_lane and extension_success_rate >= 0.45:
                tp_extension_bias += 0.03
            if runner_lane and trail_giveback_r >= 0.35 and close_winners_score < 0.55:
                early_protect_r += 0.02
            if _safe_float(management_feedback.get("missed_runner_gap_r"), 0.0) >= 0.55:
                trail_backoff_r += 0.05
                tp_extension_bias += 0.05
            if slippage_quality_score <= 0.38:
                early_protect_r -= 0.03
                stall_exit_bias += 0.08
            reentry_bias = clamp(reentry_priority + (0.10 if runner_lane else 0.0) + (0.08 if symbol_key in {"XAUUSD", "BTCUSD"} else 0.0), 0.25, 1.0)
            management_directives = {
                "early_protect_r": float(clamp(early_protect_r, 0.15, 0.55)),
                "trail_cadence_seconds": int(max(15, min(90, trail_cadence_seconds))),
                "trail_backoff_r": float(clamp(trail_backoff_r, 0.18, 0.90)),
                "tp_extension_bias": float(clamp(tp_extension_bias, 0.05, 0.45)),
                "reentry_bias": float(reentry_bias),
                "stall_exit_bias": float(clamp(stall_exit_bias, 0.10, 0.80)),
                "runner_lane": bool(runner_lane),
                "stall_lane": bool(stall_lane),
                "state_machine_enabled": True,
                "no_loosen_after_protected": True,
                "close_winners_score": float(close_winners_score),
                "extension_success_rate": float(extension_success_rate),
                "trail_giveback_r": float(trail_giveback_r),
            }
            if symbol_key == "XAUUSD" and xau_prime_session:
                if weak_focus or expectancy_r < 0.0 or win_rate < 0.55:
                    management_directives["early_protect_r"] = float(clamp(_safe_float(management_directives.get("early_protect_r"), 0.18) - 0.03, 0.15, 0.24))
                    management_directives["trail_cadence_seconds"] = int(max(8, int(management_directives.get("trail_cadence_seconds", 12) or 12) - 2))
                    management_directives["tp_extension_bias"] = float(clamp(_safe_float(management_directives.get("tp_extension_bias"), 0.18) - 0.04, 0.12, 0.32))
                    management_directives["stall_exit_bias"] = float(clamp(_safe_float(management_directives.get("stall_exit_bias"), 0.30) + 0.10, 0.10, 0.80))
                elif proof_lane_ready or hot_hand_active:
                    management_directives["trail_cadence_seconds"] = int(max(8, int(management_directives.get("trail_cadence_seconds", 12) or 12) - 2))
                    management_directives["tp_extension_bias"] = float(clamp(_safe_float(management_directives.get("tp_extension_bias"), 0.24) + 0.03, 0.05, 0.45))
            session_bankroll_bias = clamp(
                1.0
                + (0.12 if hot_hand_active else 0.0)
                + (0.10 * session_heat_score)
                + (0.08 if profit_recycle_active and expectancy_r >= 0.0 else 0.0)
                + (0.05 if symbol_key in {"XAUUSD", "BTCUSD"} and current_session_key in session_focus else 0.0),
                0.85,
                1.35,
            )
            lane_budget_share_hint = clamp(
                0.12
                + (0.10 if symbol_key in {"XAUUSD", "BTCUSD"} else 0.0)
                + (0.08 if hot_hand_active else 0.0)
                + (0.06 if session_heat_score >= 0.60 else 0.0)
                + (0.10 * max(0.0, lane_expectancy_multiplier - 1.0))
                + (profit_recycle_boost * 0.35),
                0.10,
                0.45,
            )
            frequency_directives = {
                "catchup_pressure": float(
                    clamp(
                        0.25
                        + (0.18 if bool(goal_state.get("lagging_short_goal")) else 0.0)
                        + (0.10 if symbol_key in {"XAUUSD", "BTCUSD"} else 0.0)
                        + (0.10 if asia_native_pair and current_session_key in {"SYDNEY", "TOKYO"} else 0.0)
                        + (0.08 if hour_expectancy_score >= 0.68 else 0.0)
                        + trajectory_catchup
                        + (_safe_float(local_summary.get("undertrade_pressure"), 0.0) * 0.18)
                        - (0.10 if weak_focus else 0.0),
                        0.0,
                        0.92,
                    )
                ),
                "soft_burst_target_10m": int(
                    2
                    if xau_asia_session
                    else 8
                    if symbol_key == "XAUUSD"
                    else 6
                    if symbol_key == "BTCUSD"
                    else 6
                    if asia_native_pair and current_session_key in {"SYDNEY", "TOKYO"}
                    else 4
                ),
                "session_native_priority": bool(symbol_key in {"XAUUSD", "BTCUSD"} or len(session_focus) > 0),
                "quota_boost_allowed": bool(
                    (not weak_focus or runner_lane)
                    and (
                        lagging_soft_target
                        or _safe_float(local_summary.get("undertrade_pressure"), 0.0) >= 0.20
                        or symbol_key in {"XAUUSD", "BTCUSD"}
                        or (asia_native_pair and current_session_key in {"SYDNEY", "TOKYO"})
                    )
                ),
                "aggressive_reentry_enabled": bool(reentry_bias >= 0.55 or lagging_soft_target),
                "undertrade_fix_mode": bool(local_summary.get("undertrade_fix_mode", False)),
                "idle_lane_recovery_active": bool(lane_idle_today),
            }
            if (
                lane_idle_today
                and _safe_float(local_summary.get("undertrade_pressure"), 0.0) >= 0.12
                and (
                    symbol_key in {"XAUUSD", "BTCUSD"}
                    or (asia_native_pair and current_session_key in {"SYDNEY", "TOKYO"})
                    or (current_session_key and current_session_key in session_focus)
                )
                and not xau_asia_session
            ):
                frequency_directives["catchup_pressure"] = float(
                    clamp(float(frequency_directives.get("catchup_pressure", 0.0) or 0.0) + 0.12, 0.0, 0.95)
                )
                frequency_directives["quota_boost_allowed"] = True
                frequency_directives["aggressive_reentry_enabled"] = True
                frequency_directives["undertrade_fix_mode"] = True
                frequency_directives["soft_burst_target_10m"] = max(
                    int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
                    8 if symbol_key == "XAUUSD" and xau_prime_session else 7 if symbol_key == "BTCUSD" else 5,
                )
            if hot_hand_active and not xau_asia_session:
                frequency_directives["catchup_pressure"] = float(
                    clamp(float(frequency_directives.get("catchup_pressure", 0.0) or 0.0) + 0.05, 0.0, 0.95)
                )
                frequency_directives["quota_boost_allowed"] = True
                frequency_directives["aggressive_reentry_enabled"] = True
                frequency_directives["soft_burst_target_10m"] = max(
                    int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
                    7 if symbol_key in {"XAUUSD", "BTCUSD"} else 5,
                )
            if profit_recycle_active and not xau_asia_session and expectancy_r >= 0.0:
                frequency_directives["soft_burst_target_10m"] = max(
                    int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
                    9 if symbol_key == "XAUUSD" and xau_prime_session else 7 if symbol_key == "BTCUSD" else 5,
                )
            if proof_lane_ready and not xau_asia_session:
                frequency_directives["catchup_pressure"] = float(
                    clamp(
                        float(frequency_directives.get("catchup_pressure", 0.0) or 0.0)
                        + 0.10
                        + (0.04 if symbol_key in {"XAUUSD", "BTCUSD"} else 0.0)
                        + (0.03 if good_days_last_5 >= 5 else 0.0),
                        0.0,
                        0.95,
                    )
                )
                frequency_directives["quota_boost_allowed"] = True
                frequency_directives["aggressive_reentry_enabled"] = True
                if xau_prime_session:
                    frequency_directives["soft_burst_target_10m"] = max(
                        10,
                        int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
                    )
                elif symbol_key == "BTCUSD":
                    frequency_directives["soft_burst_target_10m"] = max(
                        8,
                        int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
                    )
                elif asia_native_pair and current_session_key in {"SYDNEY", "TOKYO"}:
                    frequency_directives["soft_burst_target_10m"] = max(
                        5,
                        int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
                    )
                else:
                    frequency_directives["soft_burst_target_10m"] = max(
                        5,
                        int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
                    )
                if symbol_key in {"XAUUSD", "BTCUSD"} or (asia_native_pair and current_session_key in {"SYDNEY", "TOKYO"}):
                    frequency_directives["undertrade_fix_mode"] = True
            if xau_asia_session:
                frequency_directives["catchup_pressure"] = float(
                    min(float(frequency_directives.get("catchup_pressure", 0.0) or 0.0), 0.20)
                )
                frequency_directives["soft_burst_target_10m"] = 2
                frequency_directives["quota_boost_allowed"] = False
                frequency_directives["aggressive_reentry_enabled"] = False
                frequency_directives["undertrade_fix_mode"] = False
                management_directives["early_protect_r"] = float(clamp(management_directives["early_protect_r"], 0.18, 0.30))
                management_directives["trail_cadence_seconds"] = int(max(25, int(management_directives["trail_cadence_seconds"])))
                management_directives["trail_backoff_r"] = float(clamp(management_directives["trail_backoff_r"], 0.26, 0.38))
                management_directives["tp_extension_bias"] = float(clamp(management_directives["tp_extension_bias"], 0.05, 0.18))
            elif xau_prime_session:
                frequency_directives["soft_burst_target_10m"] = max(
                    8,
                    int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
                )
                if proof_lane_ready or hot_hand_active:
                    frequency_directives["soft_burst_target_10m"] = max(
                        10,
                        int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
                    )
                    frequency_directives["quota_boost_allowed"] = True
                    frequency_directives["aggressive_reentry_enabled"] = True
            if symbol_key in {"USOIL", "NAS100"} and news_confidence >= 0.65 and news_direction in {"bullish", "bearish", "risk_on", "risk_off"}:
                trade_horizon_bias = "daytrade"
                frequency_directives["catchup_pressure"] = float(
                    clamp(float(frequency_directives.get("catchup_pressure", 0.0) or 0.0) + 0.10, 0.0, 0.95)
                )
                frequency_directives["soft_burst_target_10m"] = max(
                    int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
                    5 if current_session_key in {"LONDON", "OVERLAP", "NEW_YORK"} else 4,
                )
                frequency_directives["quota_boost_allowed"] = True
                frequency_directives["undertrade_fix_mode"] = True
                if expectancy_r >= 0.0 or hour_expectancy_score >= 0.50:
                    frequency_directives["aggressive_reentry_enabled"] = True
            blocker_relaxation_state = {
                "temporary_block_timeout_hours": 6,
                "sydney_reset_enabled": True,
                "pair_strategy_cooldown_minutes": 60,
                "bridge_pause_timeout_minutes": 15,
                "broker_reject_quarantine_minutes": max(5, 10 if broker_reject_streak >= 2 else 5),
            }
            if lane_state_value == "attack":
                frequency_directives["soft_burst_target_10m"] = max(
                    int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
                    10 if symbol_key == "XAUUSD" else 7,
                )
                frequency_directives["quota_boost_allowed"] = True
            elif lane_state_value == "degrade":
                frequency_directives["soft_burst_target_10m"] = max(1, int(frequency_directives.get("soft_burst_target_10m", 0) or 0) - 3)
                blocker_relaxation_state["pair_strategy_cooldown_minutes"] = 30
            elif lane_shadow_only:
                frequency_directives["soft_burst_target_10m"] = 1
                frequency_directives["quota_boost_allowed"] = False
                frequency_directives["aggressive_reentry_enabled"] = False
            if lane_recent_edge_broken or lane_short_edge_broken:
                frequency_directives["aggressive_reentry_enabled"] = False
            directives[symbol_key] = {
                "trade_horizon_bias": trade_horizon_bias,
                "session_focus": list(session_focus),
                "aggression_multiplier": float(aggression_multiplier),
                "min_confluence_override": float(min_confluence_override),
                "reentry_priority": float(reentry_priority),
                "weak_focus": bool(weak_focus),
                "news_bias_direction": news_direction,
                "news_confidence": float(news_confidence),
                "history_seed_ready": bool(history_seed_ready),
                "hot_hand_active": bool(hot_hand_active),
                "hot_hand_score": float(hot_hand_score),
                "hot_hand_session_streak": int(hot_hand_session_streak),
                "session_bankroll_bias": float(session_bankroll_bias),
                "lane_budget_share_hint": float(lane_budget_share_hint),
                "profit_recycle_active": bool(profit_recycle_active),
                "profit_recycle_boost": float(profit_recycle_boost),
                "slippage_regime": "clean" if slippage_quality_score >= 0.72 else "rough" if slippage_quality_score <= 0.38 else "mixed",
                "slippage_quality_score": float(slippage_quality_score),
                "broker_reject_risk": "high" if broker_reject_streak >= 2 else "normal",
                "broker_reject_streak": int(broker_reject_streak),
                "opportunity_capture_gap_r": float(opportunity_capture_gap),
                "management_quality_score": float(management_quality_score),
                "shadow_experiment_active": bool(shadow_experiment_active),
                "hour_expectancy_score": float(hour_expectancy_score),
                "lane_expectancy_multiplier": float(lane_expectancy_multiplier),
                "strong_hours_sydney": list(strong_hours_sydney),
                "weak_hours_sydney": list(weak_hours_sydney),
                "lane_mfe_median_r": float(lane_mfe_median_r),
                "lane_mae_median_r": float(lane_mae_median_r),
                "lane_capture_efficiency": float(lane_capture_efficiency),
                "close_winners_score": float(close_winners_score),
                "tp_extension_success_rate": float(extension_success_rate),
                "trail_giveback_r": float(trail_giveback_r),
                "management_feedback_summary": dict(management_feedback),
                "trajectory_stats": dict(trajectory_stats),
                "management_directives": management_directives,
                "frequency_directives": frequency_directives,
                "blocker_relaxation_state": blocker_relaxation_state,
                "lane_state_machine": dict(lane_state_machine),
                "walk_forward_scorecards": dict(walk_forward_scorecards),
                "loss_attribution_summary": dict(loss_attribution_summary),
                "shadow_challenger_pool": dict(shadow_challenger_pool),
                "execution_quality_directives": {
                    "state": str(execution_quality_gate.get("state") or "MIXED"),
                    "quality_score": float(execution_quality_gate.get("quality_score", 0.0) or 0.0),
                    "blocked": bool(execution_quality_gate.get("blocked", False)),
                    "spread_ratio": float(execution_quality_gate.get("spread_ratio", 0.0) or 0.0),
                    "stop_sanity": float(execution_quality_gate.get("stop_sanity", 0.0) or 0.0),
                },
                "profit_budget_aggression_allowed": bool(profit_recycle_active and expectancy_r >= 0.0),
                "gpt_hybrid_advisory": {
                    "enabled": False,
                    "conviction": 0.0,
                    "direction_bias": "",
                    "setup_bias": "",
                    "threshold_delta": 0.0,
                    "aggression_delta": 0.0,
                    "session_focus": [],
                    "session_match": False,
                    "reason": "",
                },
                "targeted_backtest_probe": {
                    "enabled": bool(history_seed_ready),
                    "windows": ["M15", "H1", "D1"],
                    "reason": "weak_lane_review" if weak_focus else "maintenance_review",
                },
            }
        return directives

    def _build_meeting_packet(
        self,
        *,
        now: datetime,
        local_summary: dict[str, Any],
        learning_status: dict[str, Any],
        projection: dict[str, float],
        goal_state: dict[str, Any],
        runtime_state: dict[str, Any],
        pair_directives: dict[str, dict[str, Any]],
        current_session_name: str = "",
        offline_review: OfflineGPTReview | None = None,
    ) -> dict[str, Any]:
        return {
            "generated_at": now.isoformat(),
            "meeting_mode": "local_live_offline_gpt",
            "offline_gpt_role": "advisory_only",
            "current_session_name": str(current_session_name or "").upper(),
            "recent_trade_review": {
                "reviewed_trade_count": int(local_summary.get("reviewed_trade_count", 0) or 0),
                "win_rate": float(local_summary.get("win_rate", 0.0) or 0.0),
                "expectancy_r": float(local_summary.get("expectancy_r", 0.0) or 0.0),
                "profit_factor": float(local_summary.get("profit_factor", 0.0) or 0.0),
            },
            "market_history_status": {
                "seed_status": str(learning_status.get("last_market_history_seed_status") or ""),
                "seed_samples": int(learning_status.get("last_market_history_seed_samples") or 0),
                "backfill_status": str(learning_status.get("last_market_history_backfill_status") or ""),
                "backfill_files": int(learning_status.get("last_market_history_backfill_files") or 0),
            },
            "pair_stats": dict(local_summary.get("pair_stats") or {}),
            "pair_walk_forward_scorecards": dict(local_summary.get("pair_walk_forward_scorecards") or {}),
            "pair_loss_attribution": dict(local_summary.get("pair_loss_attribution") or {}),
            "pair_lane_state": dict(local_summary.get("pair_lane_state") or {}),
            "weak_pair_lanes": list(local_summary.get("weak_pair_lanes") or []),
            "missed_opportunity_summary": dict(local_summary.get("missed_opportunity_summary") or {}),
            "news_regime_summary": dict(local_summary.get("news_regime_summary") or {}),
            "proof_window": dict(local_summary.get("proof_window") or {}),
            "recent_lane_session_summary": dict(local_summary.get("recent_lane_session_summary") or {}),
            "lane_execution_summary": {
                str(symbol): {
                    "slippage_regime": str(directive.get("slippage_regime") or "mixed"),
                    "broker_reject_risk": str(directive.get("broker_reject_risk") or "normal"),
                    "opportunity_capture_gap_r": float(directive.get("opportunity_capture_gap_r") or 0.0),
                    "management_quality_score": float(directive.get("management_quality_score") or 0.0),
                    "close_winners_score": float(directive.get("close_winners_score") or 0.0),
                    "tp_extension_success_rate": float(directive.get("tp_extension_success_rate") or 0.0),
                    "trail_giveback_r": float(directive.get("trail_giveback_r") or 0.0),
                    "hot_hand_active": bool(directive.get("hot_hand_active", False)),
                    "hot_hand_score": float(directive.get("hot_hand_score") or 0.0),
                    "session_bankroll_bias": float(directive.get("session_bankroll_bias") or 1.0),
                    "shadow_experiment_active": bool(directive.get("shadow_experiment_active", False)),
                    "lane_state_machine": dict(directive.get("lane_state_machine") or {}),
                    "loss_attribution_summary": dict(directive.get("loss_attribution_summary") or {}),
                    "execution_quality_directives": dict(directive.get("execution_quality_directives") or {}),
                    "shadow_challenger_pool": dict(directive.get("shadow_challenger_pool") or {}),
                    "hour_expectancy_score": float(directive.get("hour_expectancy_score") or 0.0),
                    "lane_expectancy_multiplier": float(directive.get("lane_expectancy_multiplier") or 1.0),
                    "management_directives": dict(directive.get("management_directives") or {}),
                    "frequency_directives": dict(directive.get("frequency_directives") or {}),
                    "blocker_relaxation_state": dict(directive.get("blocker_relaxation_state") or {}),
                }
                for symbol, directive in pair_directives.items()
            },
            "hour_expectancy_matrix": dict(local_summary.get("lane_hour_expectancy_matrix") or {}),
            "symbol_hour_expectancy_summary": dict(local_summary.get("symbol_hour_expectancy_summary") or {}),
            "no_trade_scoreboard": dict(runtime_state.get("no_trade_scoreboard") or {}),
            "active_blockers": dict(runtime_state.get("active_blockers") or {}),
            "management_stats": dict(runtime_state.get("management_stats") or {}),
            "xau_btc_trajectory_stats": dict(runtime_state.get("xau_btc_trajectory_stats") or {}),
            "live_news_coverage_summary": dict(runtime_state.get("news_coverage_summary") or {}),
            "self_heal_status": dict(runtime_state.get("self_heal_status") or {}),
            "management_feedback_summary": dict(local_summary.get("management_feedback_summary") or {}),
            "profit_recycle_state": {
                "active": bool(local_summary.get("profit_recycle_active", False)),
                "boost": float(local_summary.get("profit_recycle_boost", 0.0) or 0.0),
            },
            "profit_budget_state": {
                "base_capital_mode": True,
                "profit_aggression_enabled": bool(local_summary.get("profit_recycle_active", False)),
                "profit_recycle_boost": float(local_summary.get("profit_recycle_boost", 0.0) or 0.0),
            },
            "daily_trade_goal_state": {
                "current_day_trade_count": int(local_summary.get("current_day_trade_count", 0) or 0),
                "daily_trade_target": int(local_summary.get("daily_trade_target", 30) or 30),
                "undertrade_fix_mode": bool(local_summary.get("undertrade_fix_mode", False)),
                "undertrade_pressure": float(local_summary.get("undertrade_pressure", 0.0) or 0.0),
            },
            "trajectory_gap": {
                "days_to_100k_at_current_rate": float(projection.get("days_to_100k_at_current_rate", 0.0) or 0.0),
                "days_to_1m_at_current_rate": float(projection.get("days_to_1m_at_current_rate", 0.0) or 0.0),
                "lagging_short_goal": bool(goal_state.get("lagging_short_goal")),
                "lagging_medium_goal": bool(goal_state.get("lagging_medium_goal")),
            },
            "runtime_news_unknown_symbols": list(runtime_state.get("news_unknown_symbols") or []),
            "hybrid_pair_ideas": list((offline_review.hybrid_pair_ideas if offline_review is not None else []) or []),
            "next_cycle_directives": dict(pair_directives),
        }

    def _trajectory_projection(
        self,
        *,
        local_summary: dict[str, Any],
        account_state: dict[str, Any],
    ) -> dict[str, float]:
        equity = max(1.0, _safe_float(account_state.get("equity"), _safe_float(account_state.get("balance"), 0.0)))
        expectancy_r = max(-2.0, min(4.0, _safe_float(local_summary.get("expectancy_r"), 0.0)))
        win_rate = clamp(_safe_float(local_summary.get("win_rate"), 0.0), 0.0, 1.0)
        base_growth_factor = max(0.90, 1.0 + (expectancy_r * 0.04) + ((win_rate - 0.5) * 0.12))
        projection_56d = equity * (base_growth_factor ** 56)
        projection_30d = equity * (base_growth_factor ** 30)
        projection_90d = equity * (base_growth_factor ** 90)
        projection_365d = equity * (base_growth_factor ** 365)
        days_to_100k = self._days_to_target(
            current_equity=equity,
            target_equity=self.short_goal_equity,
            daily_growth_factor=base_growth_factor,
        )
        days_to_1m = self._days_to_target(
            current_equity=equity,
            target_equity=self.medium_goal_equity,
            daily_growth_factor=base_growth_factor,
        )
        return {
            "current_equity": float(equity),
            "daily_growth_factor": float(base_growth_factor),
            "projection_56d": float(projection_56d),
            "projection_30d": float(projection_30d),
            "projection_90d": float(projection_90d),
            "projection_365d": float(projection_365d),
            "gap_to_short_goal": float(max(0.0, self.short_goal_equity - projection_90d)),
            "gap_to_medium_goal": float(max(0.0, self.medium_goal_equity - projection_365d)),
            "days_to_100k_at_current_rate": float(days_to_100k),
            "days_to_1m_at_current_rate": float(days_to_1m),
        }

    def _goal_state(self, *, projection: dict[str, float]) -> dict[str, Any]:
        short_target_days = 56
        medium_target_days = 365
        projection_56d = _safe_float(projection.get("projection_56d"), 0.0)
        projection_365d = _safe_float(projection.get("projection_365d"), 0.0)
        return {
            "current_equity": float(_safe_float(projection.get("current_equity"), 0.0)),
            "short_goal_equity": float(self.short_goal_equity),
            "medium_goal_equity": float(self.medium_goal_equity),
            "short_goal_days": int(short_target_days),
            "medium_goal_days": int(medium_target_days),
            "projection_56d": float(projection_56d),
            "projection_365d": float(projection_365d),
            "short_goal_on_track": bool(projection_56d >= self.short_goal_equity),
            "medium_goal_on_track": bool(projection_365d >= self.medium_goal_equity),
            "lagging_short_goal": bool(projection_56d < self.short_goal_equity),
            "lagging_medium_goal": bool(projection_365d < self.medium_goal_equity),
        }

    def _build_autonomy_state(
        self,
        *,
        now: datetime,
        local_summary: dict[str, Any],
        learning_status: dict[str, Any],
        runtime_state: dict[str, Any],
        account_state: dict[str, Any],
        projection: dict[str, float],
        goal_state: dict[str, Any],
    ) -> dict[str, Any]:
        no_trade_minutes = max(0.0, _safe_float(runtime_state.get("no_trade_minutes"), 0.0))
        rolling_drawdown_pct = max(
            0.0,
            _safe_float(runtime_state.get("rolling_drawdown_pct"), _safe_float(account_state.get("rolling_drawdown_pct"), 0.0)),
        )
        absolute_drawdown_pct = max(
            0.0,
            _safe_float(runtime_state.get("absolute_drawdown_pct"), _safe_float(account_state.get("absolute_drawdown_pct"), 0.0)),
        )
        learner_status = str(
            runtime_state.get("learner_status")
            or learning_status.get("last_maintenance_status")
            or ""
        ).strip()
        bridge_state = dict(runtime_state.get("bridge_singleton_status") or {})
        news_unknown_symbols = [
            str(item).upper()
            for item in runtime_state.get("news_unknown_symbols", [])
            if str(item).strip()
        ]
        spread_spike_symbols = [
            str(item).upper()
            for item in runtime_state.get("spread_spike_symbols", [])
            if str(item).strip()
        ]
        anomaly_reasons: list[str] = []
        actions: list[dict[str, Any]] = []
        active_failures: dict[str, int] = {}
        previous_failures = dict(self._brain_state.get("self_heal_failure_counts") or {})

        def _flag(code: str, reason: str) -> None:
            anomaly_reasons.append(f"{code}:{reason}")
            active_failures[code] = int(previous_failures.get(code, 0) or 0) + 1

        bridge_unhealthy = (
            not bool(bridge_state.get("singleton_enforced", False))
            or bool(bridge_state.get("listener_conflict", False))
            or int(bridge_state.get("owner_pid", 0) or 0) <= 0
        )
        if no_trade_minutes >= 120.0:
            _flag("no_trades_2h", f"no_trades_for_{int(no_trade_minutes)}m")
            actions.extend(
                [
                    {
                        "kind": "loosen_quota_proven_winners",
                        "apply_now": True,
                        "severity": "warning",
                        "reason": f"trade_flow_stalled_{int(no_trade_minutes)}m",
                    },
                    {
                        "kind": "reload_latest_checkpoint",
                        "apply_now": True,
                        "severity": "warning",
                        "reason": "stalled_trade_flow_checkpoint_refresh",
                    },
                ]
            )
        if rolling_drawdown_pct >= 0.06:
            _flag("drawdown_over_6pct", f"rolling_drawdown_{rolling_drawdown_pct:.4f}")
            actions.append(
                {
                    "kind": "tighten_risk_mode",
                    "apply_now": True,
                    "severity": "warning",
                    "reason": f"rolling_drawdown_{rolling_drawdown_pct:.4f}",
                }
            )
        if absolute_drawdown_pct >= 0.08:
            _flag("drawdown_over_8pct", f"absolute_drawdown_{absolute_drawdown_pct:.4f}")
            actions.append(
                {
                    "kind": "emergency_sydney_reset",
                    "apply_now": True,
                    "severity": "critical",
                    "reason": f"absolute_drawdown_{absolute_drawdown_pct:.4f}",
                }
            )
        learner_stuck = learner_status.lower() in {
            "insufficient_class_balance",
            "hold_insufficient_class_balance",
            "feature_mismatch_reset",
            "stuck",
        }
        if learner_stuck:
            _flag("learner_stuck", learner_status or "learner_stuck")
            actions.extend(
                [
                    {
                        "kind": "force_local_retrain",
                        "apply_now": True,
                        "severity": "warning",
                        "reason": learner_status or "learner_stuck",
                    },
                    {
                        "kind": "schedule_sydney_temp_block_clear",
                        "apply_now": False,
                        "severity": "info",
                        "reason": learner_status or "learner_stuck",
                    },
                ]
            )
        if bridge_unhealthy:
            _flag("bridge_unhealthy", "singleton_or_listener_issue")
            actions.append(
                {
                    "kind": "restart_bridge_if_dead",
                    "apply_now": True,
                    "severity": "critical",
                    "reason": "singleton_or_listener_issue",
                }
            )
        if news_unknown_symbols:
            _flag("news_unknown", ",".join(news_unknown_symbols[:6]))
            actions.append(
                {
                    "kind": "resolve_news_state_fallback",
                    "apply_now": True,
                    "severity": "warning",
                    "reason": "news_unknown_symbols_present",
                    "symbols": list(news_unknown_symbols[:8]),
                }
            )
        if spread_spike_symbols:
            _flag("spread_spike", ",".join(spread_spike_symbols[:6]))
            actions.append(
                {
                    "kind": "pause_pair_spread_spike",
                    "apply_now": True,
                    "severity": "warning",
                    "reason": "spread_spike_symbols_present",
                    "symbols": list(spread_spike_symbols[:8]),
                }
            )
        if bool(goal_state.get("lagging_short_goal")):
            anomaly_reasons.append("trajectory_lag:short_goal_projection_below_target")
            actions.append(
                {
                    "kind": "catchup_burst_mode",
                    "apply_now": True,
                    "severity": "info",
                    "reason": "short_goal_projection_below_target",
                }
            )

        critical_failure_codes = {"drawdown_over_6pct", "drawdown_over_8pct", "bridge_unhealthy", "news_unknown", "spread_spike"}
        repair_failed_twice = any(
            code in critical_failure_codes and count >= 2
            for code, count in active_failures.items()
        )
        if repair_failed_twice:
            actions.append(
                {
                    "kind": "reduce_risk_50pct",
                    "apply_now": True,
                    "severity": "critical",
                    "reason": "repair_failed_twice",
                }
            )

        deduped_actions: list[dict[str, Any]] = []
        seen_actions: set[str] = set()
        for action in actions:
            serialized = json.dumps(action, sort_keys=True)
            if serialized in seen_actions:
                continue
            seen_actions.add(serialized)
            deduped_actions.append(dict(action))

        return {
            "generated_at": now.isoformat(),
            "anomaly_active": bool(active_failures),
            "anomaly_reasons": list(dict.fromkeys(anomaly_reasons)),
            "failure_counts": active_failures,
            "no_trade_minutes": float(no_trade_minutes),
            "rolling_drawdown_pct": float(rolling_drawdown_pct),
            "absolute_drawdown_pct": float(absolute_drawdown_pct),
            "bridge_unhealthy": bool(bridge_unhealthy),
            "learner_status": learner_status,
            "news_unknown_symbols": news_unknown_symbols,
            "spread_spike_symbols": spread_spike_symbols,
            "repair_failed_twice": bool(repair_failed_twice),
            "self_heal_actions": deduped_actions,
        }

    def _build_shadow_strategy_variants(
        self,
        *,
        now: datetime,
        local_summary: dict[str, Any],
        optimizer_summary: dict[str, Any],
        runtime_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        weak_pairs = [str(item).upper() for item in local_summary.get("weak_pair_focus", []) if str(item).strip()]
        shadow_focus = [
            str(item).upper()
            for item in (optimizer_summary.get("performance_buckets") or {}).keys()
            if str(item).strip()
        ]
        runtime_symbols = [str(item).upper() for item in runtime_state.get("symbols", []) if str(item).strip()]
        seed_symbols = list(dict.fromkeys(weak_pairs + shadow_focus + runtime_symbols))
        if not seed_symbols:
            seed_symbols = ["XAUUSD", "BTCUSD"]
        hot_symbols = {
            str(symbol).upper()
            for symbol, payload in (optimizer_summary.get("performance_buckets") or {}).items()
            if isinstance(payload, dict)
            and (
                _safe_float(payload.get("expectancy_r"), 0.0) >= 0.12
                or _safe_float(payload.get("win_rate"), 0.0) >= 0.58
            )
        }
        variant_count = self.shadow_hot_variants if any(symbol in hot_symbols for symbol in seed_symbols) else self.shadow_default_variants
        session_cycle = ["TOKYO", "LONDON", "OVERLAP", "NEW_YORK", "SYDNEY"]
        variants: list[dict[str, Any]] = []
        for index in range(variant_count):
            symbol_key = seed_symbols[index % len(seed_symbols)]
            session_name = session_cycle[index % len(session_cycle)]
            score_hint = round(
                clamp(
                    0.45
                    + (0.08 * (index % 3))
                    + (0.05 if symbol_key in weak_pairs else 0.0)
                    + (0.04 if symbol_key in hot_symbols else 0.0),
                    0.0,
                    1.0,
                ),
                3,
            )
            performance_hint = {
                "expectancy_r": round((score_hint - 0.45) * 1.6, 3),
                "profit_factor": round(1.0 + max(0.0, score_hint - 0.48), 3),
                "slippage_quality_score": round(clamp(0.62 + (0.03 * (index % 5)), 0.0, 1.0), 3),
            }
            promotion = score_shadow_variant(
                {"variant_id": f"{symbol_key}_SHADOW_{index+1}", "score_hint": score_hint},
                performance=performance_hint,
                promotion_threshold=self.shadow_promotion_threshold,
            )
            variants.append(
                {
                    "variant_id": str(promotion.variant_id),
                    "generated_at": now.isoformat(),
                    "symbol": symbol_key,
                    "session": session_name,
                    "status": "shadow_candidate",
                    "genome": {
                        "burst_count": 2 + (index % 4),
                        "spacing_multiplier": round(0.82 + (0.04 * (index % 5)), 3),
                        "add_threshold": round(0.35 + (0.03 * (index % 4)), 3),
                        "reclaim_filter": round(0.48 + (0.02 * (index % 5)), 3),
                        "exit_aggression": round(0.55 + (0.03 * (index % 5)), 3),
                        "session_quota": 4 + (index % 4),
                    },
                    "score_hint": float(score_hint),
                    "promotion_score": float(promotion.promotion_score),
                    "expectancy_r": float(promotion.expectancy_r),
                    "profit_factor": float(promotion.profit_factor),
                    "slippage_adjusted_score": float(promotion.slippage_adjusted_score),
                    "promoted_candidate": bool(promotion.promoted),
                    "rationale": (
                        "local_ga_shadow_variant_for_weak_pair"
                        if symbol_key in weak_pairs
                        else "local_ga_shadow_variant_for_expansion"
                    ),
                }
            )
        return variants

    def _build_promotion_bundle(
        self,
        *,
        now: datetime,
        local_summary: dict[str, Any],
        learning_status: dict[str, Any],
        optimizer_summary: dict[str, Any],
        projection: dict[str, float],
        goal_state: dict[str, Any],
        anomaly_state: dict[str, Any],
        shadow_strategy_variants: list[dict[str, Any]],
        runtime_state: dict[str, Any],
        pair_directives: dict[str, dict[str, Any]],
        meeting_packet: dict[str, Any],
    ) -> LearningPromotionBundle:
        win_rate = clamp(_safe_float(local_summary.get("win_rate"), 0.0), 0.0, 1.0)
        expectancy_r = _safe_float(local_summary.get("expectancy_r"), 0.0)
        profit_factor = max(0.0, _safe_float(local_summary.get("profit_factor"), 0.0))
        winner_loss_ratio = max(0.0, _safe_float(local_summary.get("winner_loss_ratio"), 0.0))
        proof_window = dict(local_summary.get("proof_window") or {})
        proof_ready_5d = bool(proof_window.get("proof_ready_5d", False))
        good_days_last_5 = max(0, _safe_int(proof_window.get("good_days_last_5"), 0))
        daily_rows = list(proof_window.get("daily_rows") or [])
        latest_day_row = dict(daily_rows[-1] or {}) if daily_rows else {}
        latest_day_green = bool(latest_day_row.get("good_day", False))
        latest_day_expectancy = _safe_float(latest_day_row.get("expectancy_r"), 0.0)
        latest_day_trade_count = max(0, _safe_int(latest_day_row.get("trade_count"), 0))
        gap_to_goal = max(0.0, _safe_float(projection.get("gap_to_short_goal"), 0.0))
        current_equity = max(1.0, _safe_float(projection.get("current_equity"), 0.0))
        healthy_streak = win_rate >= 0.60 and expectancy_r > 0.0 and profit_factor >= 1.05
        undertrade_pressure = clamp(_safe_float(local_summary.get("undertrade_pressure"), 0.0), 0.0, 1.0)
        hot_hand_pairs = [
            str(symbol)
            for symbol, directive in (pair_directives or {}).items()
            if bool((directive or {}).get("hot_hand_active", False))
        ]
        hot_hand_active = bool(hot_hand_pairs)
        proof_scaling_ready = bool(
            proof_ready_5d
            and healthy_streak
            and (winner_loss_ratio >= 1.1 or good_days_last_5 >= 5)
        )
        profit_recycle_active = bool(
            latest_day_green
            and latest_day_trade_count >= 3
            and latest_day_expectancy >= 0.0
            and healthy_streak
        )
        profit_recycle_boost = clamp(
            (0.05 if profit_recycle_active else 0.0)
            + (0.04 if hot_hand_active else 0.0)
            + (0.05 if latest_day_expectancy >= 0.35 else 0.0),
            0.0,
            0.18,
        )
        if current_equity < 150.0:
            risk_pct_target = 0.01
        elif current_equity < 300.0:
            risk_pct_target = 0.015 if healthy_streak else 0.0125
        elif current_equity >= 1500.0 and proof_scaling_ready:
            risk_pct_target = 0.04
        elif current_equity >= 500.0 and healthy_streak and good_days_last_5 >= 3:
            risk_pct_target = 0.035
        elif proof_scaling_ready:
            risk_pct_target = 0.025
        elif current_equity >= 1000.0 and healthy_streak and winner_loss_ratio >= 1.2:
            risk_pct_target = 0.025
        elif current_equity >= 500.0 and healthy_streak:
            risk_pct_target = 0.02
        else:
            risk_pct_target = 0.015
        if profit_recycle_active:
            risk_pct_target = clamp(risk_pct_target + 0.0025, 0.005, 0.04)
        if hot_hand_active and proof_scaling_ready:
            risk_pct_target = clamp(risk_pct_target + 0.0015, 0.005, 0.04)
        catchup_pressure = clamp(
            (gap_to_goal / max(self.short_goal_equity, 1.0))
            + (0.20 if expectancy_r < 0.4 else 0.0)
            + (undertrade_pressure * 0.25),
            0.0,
            1.0,
        )
        lot_size_multiplier = clamp(
            1.0
            + (0.34 * catchup_pressure)
            + (0.10 if healthy_streak else 0.0)
            + (0.10 if current_equity >= 500.0 else 0.0),
            1.0,
            1.6,
        )
        if proof_ready_5d:
            lot_size_multiplier = clamp(lot_size_multiplier + 0.10, 1.0, 1.7)
        if profit_recycle_active:
            lot_size_multiplier = clamp(lot_size_multiplier + (profit_recycle_boost * 0.80), 1.0, 1.85)
        if hot_hand_active:
            lot_size_multiplier = clamp(lot_size_multiplier + 0.08, 1.0, 1.90)
        if current_equity >= 500.0 and healthy_streak and good_days_last_5 >= 3:
            lot_size_multiplier = clamp(lot_size_multiplier + 0.18, 1.0, 2.0)
        max_open_trades_multiplier = clamp(
            1.0
            + (0.45 * catchup_pressure)
            + (0.10 if healthy_streak else 0.0)
            + (0.10 if current_equity >= 500.0 else 0.0),
            1.0,
            1.6,
        )
        if proof_ready_5d:
            max_open_trades_multiplier = clamp(max_open_trades_multiplier + 0.12, 1.0, 1.8)
        if profit_recycle_active:
            max_open_trades_multiplier = clamp(max_open_trades_multiplier + (profit_recycle_boost * 0.95), 1.0, 1.95)
        if hot_hand_active:
            max_open_trades_multiplier = clamp(max_open_trades_multiplier + 0.10, 1.0, 2.00)
        if current_equity >= 500.0 and healthy_streak and good_days_last_5 >= 3:
            max_open_trades_multiplier = clamp(max_open_trades_multiplier + 0.20, 1.0, 2.00)
        risk_reduction_active = any(
            str(item.get("kind") or "") == "reduce_risk_50pct"
            for item in anomaly_state.get("self_heal_actions", [])
            if isinstance(item, dict)
        )
        if risk_reduction_active:
            risk_pct_target = clamp(risk_pct_target * 0.5, 0.005, 0.025)
            lot_size_multiplier = clamp(lot_size_multiplier * 0.85, 0.7, 1.5)
            max_open_trades_multiplier = clamp(max_open_trades_multiplier * 0.85, 0.8, 1.5)
            profit_recycle_active = False
            profit_recycle_boost = 0.0
        promoted_patterns = []
        strategies = optimizer_summary.get("strategies", {}) if isinstance(optimizer_summary.get("strategies"), dict) else {}
        for strategy_key, payload in strategies.items():
            if not isinstance(payload, dict):
                continue
            state = str(payload.get("state") or "").lower()
            if state == "promoted":
                promoted_patterns.append(str(strategy_key))
        shadow_pair_focus = [
            key
            for key, payload in sorted(
                (optimizer_summary.get("performance_buckets") or {}).items(),
                key=lambda item: _safe_float((item[1] or {}).get("expectancy_r"), 0.0),
            )[:5]
            if isinstance(payload, dict)
        ]
        runtime_symbols = [
            _symbol_key(item)
            for item in runtime_state.get("symbols", [])
            if str(item).strip()
        ]
        weak_pair_focus = [str(item).upper() for item in local_summary.get("weak_pair_focus") or [] if str(item).strip()]
        anomaly_reasons = [str(item) for item in anomaly_state.get("anomaly_reasons", []) if str(item).strip()]
        stalled_trade_flow = any(reason.startswith("no_trades_2h:") for reason in anomaly_reasons)
        if stalled_trade_flow and runtime_symbols:
            weak_pair_focus = list(dict.fromkeys(weak_pair_focus + runtime_symbols))
            shadow_pair_focus = list(dict.fromkeys(shadow_pair_focus + runtime_symbols))
        return LearningPromotionBundle(
            generated_at=now.isoformat(),
            ga_generation_id=int(learning_status.get("ga_generation_id") or 0),
            mode="local_live_offline_gpt",
            monte_carlo_min_realities=int(self.monte_carlo_min_realities),
            monte_carlo_pass_floor=float(self.monte_carlo_pass_floor),
            risk_pct_target=float(risk_pct_target),
            max_open_trades_multiplier=float(max_open_trades_multiplier),
            lot_size_multiplier=float(lot_size_multiplier),
            quota_catchup_pressure=float(catchup_pressure),
            shadow_pair_focus=list(shadow_pair_focus),
            weak_pair_focus=list(weak_pair_focus),
            promoted_patterns=list(promoted_patterns[:12]),
            trajectory_projection=dict(projection),
            goal_state=dict(goal_state),
            self_heal_actions=[
                dict(item)
                for item in anomaly_state.get("self_heal_actions", [])
                if isinstance(item, dict)
            ],
            shadow_strategy_variants=list(shadow_strategy_variants[: self.shadow_hot_variants]),
            risk_reduction_active=bool(risk_reduction_active),
            recovery_mode_active=bool(runtime_state.get("recovery_mode_active", False)),
            pair_directives=dict(pair_directives),
            meeting_packet=dict(meeting_packet),
            local_summary={
                "win_rate": float(win_rate),
                "expectancy_r": float(expectancy_r),
                "profit_factor": float(profit_factor),
                "winner_loss_ratio": float(winner_loss_ratio),
                "proof_window": {
                    "good_days_last_5": int(good_days_last_5),
                    "proof_ready_5d": bool(proof_ready_5d),
                },
                "reviewed_trade_count": int(local_summary.get("reviewed_trade_count") or 0),
                "winning_pattern_memory": list(local_summary.get("winning_pattern_memory") or []),
                "proven_reentry_queue": list(local_summary.get("proven_reentry_queue") or []),
                "anomaly_reasons": list(anomaly_state.get("anomaly_reasons") or []),
                "failure_reasons": dict(local_summary.get("failure_reasons") or {}),
                "pair_walk_forward_scorecards": dict(local_summary.get("pair_walk_forward_scorecards") or {}),
                "pair_loss_attribution": dict(local_summary.get("pair_loss_attribution") or {}),
                "pair_lane_state": dict(local_summary.get("pair_lane_state") or {}),
                "missed_opportunity_summary": dict(local_summary.get("missed_opportunity_summary") or {}),
                "news_regime_summary": dict(local_summary.get("news_regime_summary") or {}),
                "management_feedback_summary": dict(local_summary.get("management_feedback_summary") or {}),
                "current_day_trade_count": int(local_summary.get("current_day_trade_count") or 0),
                "daily_trade_target": int(local_summary.get("daily_trade_target") or 30),
                "undertrade_fix_mode": bool(local_summary.get("undertrade_fix_mode", False)),
                "undertrade_pressure": float(local_summary.get("undertrade_pressure") or 0.0),
                "profit_recycle_active": bool(profit_recycle_active),
                "profit_recycle_boost": float(profit_recycle_boost),
                "hot_hand_pairs": hot_hand_pairs[:8],
                "runtime_pair_count": int(len(runtime_state.get("symbols", []) or [])) if isinstance(runtime_state.get("symbols"), list) else 0,
            },
        )

    def _offline_gpt_review(
        self,
        *,
        now: datetime,
        weekly_prep: bool,
        local_summary: dict[str, Any],
        promotion_bundle: LearningPromotionBundle,
        projection: dict[str, float],
        learning_status: dict[str, Any],
    ) -> OfflineGPTReview:
        context = {
            "mode": "monday_prep" if weekly_prep else "four_hour_review",
            "timestamp_utc": now.isoformat(),
            "reviewed_trade_count": int(local_summary.get("reviewed_trade_count") or 0),
            "local_summary": local_summary,
            "promotion_bundle": asdict(promotion_bundle),
            "learning_status": dict(learning_status),
            "trajectory_projection": projection,
            "daily_trade_goal_state": {
                "current_day_trade_count": int(local_summary.get("current_day_trade_count") or 0),
                "daily_trade_target": int(local_summary.get("daily_trade_target") or 30),
                "undertrade_fix_mode": bool(local_summary.get("undertrade_fix_mode", False)),
                "undertrade_pressure": float(local_summary.get("undertrade_pressure") or 0.0),
            },
            "management_feedback_summary": dict(local_summary.get("management_feedback_summary") or {}),
            "goals": {
                "short_equity": float(self.short_goal_equity),
                "medium_equity": float(self.medium_goal_equity),
            },
            "required_schema": {
                "summary": "string",
                "weak_patterns": ["string"],
                "strategy_ideas": ["string"],
                "next_cycle_focus": ["string"],
                "reentry_watchlist": ["string"],
                "weekly_trade_ideas": ["string"],
                "hybrid_pair_ideas": [
                    {
                        "symbol": "string",
                        "session_focus": ["string"],
                        "setup_bias": "string",
                        "direction_bias": "string",
                        "conviction": "number",
                        "aggression_delta": "number",
                        "threshold_delta": "number",
                        "reason": "string",
                    }
                ],
            },
            "task_emphasis": (
                "undertrade_fix_mode_with_bounded_threshold_retunes"
                if bool(local_summary.get("undertrade_fix_mode", False))
                else "standard_four_hour_review"
            ),
        }
        payload, error = self._offline_gpt.offline_review(
            {
                "task": "APEX_LEARNING_BRAIN_OFFLINE_REVIEW",
                **context,
            }
        )
        if payload is None:
            return OfflineGPTReview(
                generated_at=now.isoformat(),
                status=f"offline_gpt_unavailable:{error or 'unknown'}",
                error=str(error or "offline_gpt_unavailable"),
                summary="Local brain cycle completed without GPT review.",
                next_cycle_focus=list(promotion_bundle.weak_pair_focus[:3]),
            )
        return OfflineGPTReview(
            generated_at=now.isoformat(),
            status="ok",
            summary=str(payload.get("summary") or ""),
            weak_patterns=[str(item) for item in payload.get("weak_patterns", []) if str(item).strip()],
            strategy_ideas=[str(item) for item in payload.get("strategy_ideas", []) if str(item).strip()],
            next_cycle_focus=[str(item) for item in payload.get("next_cycle_focus", []) if str(item).strip()],
            reentry_watchlist=[str(item) for item in payload.get("reentry_watchlist", []) if str(item).strip()],
            weekly_trade_ideas=[str(item) for item in payload.get("weekly_trade_ideas", []) if str(item).strip()],
            hybrid_pair_ideas=self._normalize_hybrid_pair_ideas(payload.get("hybrid_pair_ideas")),
        )

    @staticmethod
    def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return dict(default)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return dict(default)
        return payload if isinstance(payload, dict) else dict(default)

    def _load_promotion_bundle(self) -> LearningPromotionBundle:
        payload = self._load_json(self.promotion_bundle_path, {})
        if not payload:
            now_iso = utc_now().isoformat()
            return LearningPromotionBundle(
                generated_at=now_iso,
                ga_generation_id=0,
                mode="local_live_offline_gpt",
                monte_carlo_min_realities=int(self.monte_carlo_min_realities),
                monte_carlo_pass_floor=float(self.monte_carlo_pass_floor),
                risk_pct_target=0.01,
                max_open_trades_multiplier=1.0,
                lot_size_multiplier=1.0,
                quota_catchup_pressure=0.0,
                shadow_pair_focus=[],
                weak_pair_focus=[],
                promoted_patterns=[],
                trajectory_projection={},
                goal_state={},
                self_heal_actions=[],
                shadow_strategy_variants=[],
                risk_reduction_active=False,
                recovery_mode_active=False,
                pair_directives={},
                meeting_packet={},
                local_summary={},
            )
        return LearningPromotionBundle(
            generated_at=str(payload.get("generated_at") or utc_now().isoformat()),
            ga_generation_id=int(payload.get("ga_generation_id") or 0),
            mode=str(payload.get("mode") or "local_live_offline_gpt"),
            monte_carlo_min_realities=max(500, int(payload.get("monte_carlo_min_realities") or self.monte_carlo_min_realities)),
            monte_carlo_pass_floor=clamp(float(payload.get("monte_carlo_pass_floor") or self.monte_carlo_pass_floor), 0.5, 0.99),
            risk_pct_target=clamp(float(payload.get("risk_pct_target") or 0.01), 0.005, 0.05),
            max_open_trades_multiplier=clamp(float(payload.get("max_open_trades_multiplier") or 1.0), 1.0, 2.0),
            lot_size_multiplier=clamp(float(payload.get("lot_size_multiplier") or 1.0), 1.0, 2.0),
            quota_catchup_pressure=clamp(float(payload.get("quota_catchup_pressure") or 0.0), 0.0, 1.0),
            shadow_pair_focus=[str(item) for item in payload.get("shadow_pair_focus", []) if str(item).strip()],
            weak_pair_focus=[str(item) for item in payload.get("weak_pair_focus", []) if str(item).strip()],
            promoted_patterns=[str(item) for item in payload.get("promoted_patterns", []) if str(item).strip()],
            trajectory_projection=dict(payload.get("trajectory_projection") or {}),
            goal_state=dict(payload.get("goal_state") or {}),
            self_heal_actions=[
                dict(item)
                for item in payload.get("self_heal_actions", [])
                if isinstance(item, dict)
            ],
            shadow_strategy_variants=[
                dict(item)
                for item in payload.get("shadow_strategy_variants", [])
                if isinstance(item, dict)
            ],
            risk_reduction_active=bool(payload.get("risk_reduction_active", False)),
            recovery_mode_active=bool(payload.get("recovery_mode_active", False)),
            pair_directives={
                str(key).upper(): dict(value)
                for key, value in (payload.get("pair_directives") or {}).items()
                if isinstance(value, dict)
            },
            meeting_packet=dict(payload.get("meeting_packet") or {}),
            local_summary=dict(payload.get("local_summary") or {}),
        )

    @staticmethod
    def _days_to_target(
        *,
        current_equity: float,
        target_equity: float,
        daily_growth_factor: float,
    ) -> float:
        current = max(1.0, float(current_equity))
        target = max(current, float(target_equity))
        growth = float(daily_growth_factor)
        if current >= target:
            return 0.0
        if growth <= 1.0001:
            return 9999.0
        days = 0
        projected = current
        while projected < target and days < 9999:
            projected *= growth
            days += 1
        return float(days)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        ensure_parent(path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        ensure_parent(path)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
