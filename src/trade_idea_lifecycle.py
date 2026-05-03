from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from src.utils import deterministic_id

UTC = timezone.utc


IDEA_ACTIVE = "ACTIVE"
IDEA_REJECTED = "REJECTED"
IDEA_COOLDOWN = "COOLDOWN"
IDEA_ARCHIVED = "ARCHIVED"


@dataclass
class TradeIdea:
    id: str
    symbol: str
    setup_type: str
    side: str
    confidence: float
    confluence_score: float
    timestamp_created: datetime
    last_evaluated: datetime | None = None
    rejection_reason: str = ""
    evaluation_count: int = 0
    status: str = IDEA_ACTIVE
    cooldown_until: datetime | None = None
    last_structure: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeIdeaLifecycle:
    archive_minutes: int = 10
    max_rechecks_per_idea: int = 5
    max_active_ideas_per_symbol: int = 3
    recheck_seconds_default: int = 60
    recheck_seconds_by_session: dict[str, int] = field(default_factory=dict)
    cooldown_seconds_by_session: dict[str, int] = field(default_factory=dict)
    ideas: dict[str, TradeIdea] = field(default_factory=dict)

    def upsert(
        self,
        *,
        symbol: str,
        setup_type: str,
        side: str,
        confidence: float,
        confluence_score: float,
        entry_price: float,
        atr: float,
        now: datetime,
        structure: dict[str, Any],
    ) -> TradeIdea:
        idea_id = self._idea_id(symbol=symbol, setup_type=setup_type, side=side, entry_price=entry_price, atr=atr)
        idea = self.ideas.get(idea_id)
        if (
            idea is not None
            and idea.status == IDEA_ARCHIVED
            and self._allows_fast_reactivation(symbol=symbol, setup_type=setup_type)
            and idea.last_evaluated is not None
            and (now - idea.last_evaluated).total_seconds() >= 300
        ):
            idea = None
            self.ideas.pop(idea_id, None)
        if idea is None:
            self._enforce_symbol_cap(symbol=symbol)
            idea = TradeIdea(
                id=idea_id,
                symbol=symbol.upper(),
                setup_type=str(setup_type),
                side=str(side).upper(),
                confidence=float(confidence),
                confluence_score=float(confluence_score),
                timestamp_created=now,
                last_structure=dict(structure),
            )
            self.ideas[idea_id] = idea
            return idea
        idea.confidence = float(confidence)
        idea.confluence_score = float(confluence_score)
        return idea

    def can_evaluate(self, *, idea: TradeIdea, now: datetime, session_name: str, structure: dict[str, Any]) -> tuple[bool, str]:
        if idea.status == IDEA_ARCHIVED:
            return False, "archived"

        self._archive_if_expired(idea=idea, now=now)
        if idea.status == IDEA_ARCHIVED:
            return False, "archived"

        structure_changed = self._structure_changed(previous=idea.last_structure, current=structure)
        if idea.status in {IDEA_REJECTED, IDEA_COOLDOWN} and structure_changed:
            idea.status = IDEA_ACTIVE
            idea.evaluation_count = 0
            idea.rejection_reason = ""
            idea.cooldown_until = None
            idea.last_structure = dict(structure)
            return True, "reactivated"

        if idea.status in {IDEA_REJECTED, IDEA_COOLDOWN}:
            if idea.cooldown_until and now < idea.cooldown_until:
                return False, "cooldown_active"
            if idea.rejection_reason == "delivery_pending":
                return True, "active"
            interval = max(1, self._session_seconds(self.recheck_seconds_by_session, session_name, self.recheck_seconds_default))
            if idea.last_evaluated and (now - idea.last_evaluated).total_seconds() < interval:
                return False, "recheck_wait"

        return True, "active"

    def mark_evaluated(self, idea: TradeIdea, now: datetime, structure: dict[str, Any]) -> None:
        idea.last_evaluated = now
        idea.last_structure = dict(structure)

    def reject(self, *, idea: TradeIdea, reason: str, now: datetime, session_name: str, structure: dict[str, Any]) -> None:
        idea.last_evaluated = now
        idea.evaluation_count += 1
        idea.rejection_reason = str(reason)
        idea.last_structure = dict(structure)
        idea.status = IDEA_REJECTED
        cooldown_seconds = max(1, self._session_seconds(self.cooldown_seconds_by_session, session_name, 30))
        idea.cooldown_until = now + timedelta(seconds=cooldown_seconds)
        idea.status = IDEA_COOLDOWN
        self._archive_if_expired(idea=idea, now=now)

    def mark_trade_sent(self, *, idea: TradeIdea, now: datetime) -> None:
        idea.last_evaluated = now
        idea.status = IDEA_ARCHIVED
        idea.cooldown_until = None

    def mark_delivery_pending(self, *, idea: TradeIdea, now: datetime, retry_after_seconds: int = 30) -> None:
        idea.last_evaluated = now
        idea.rejection_reason = "delivery_pending"
        idea.status = IDEA_COOLDOWN
        idea.cooldown_until = now + timedelta(seconds=max(5, int(retry_after_seconds)))

    def archive_stale(self, now: datetime) -> None:
        for idea in self.ideas.values():
            self._archive_if_expired(idea=idea, now=now)

    def summary(self, symbol: str) -> dict[str, int]:
        output = {IDEA_ACTIVE: 0, IDEA_REJECTED: 0, IDEA_COOLDOWN: 0, IDEA_ARCHIVED: 0}
        normalized = str(symbol).upper()
        for idea in self.ideas.values():
            if idea.symbol != normalized:
                continue
            output[idea.status] = int(output.get(idea.status, 0)) + 1
        return output

    @staticmethod
    def build_structure_snapshot(
        *,
        row: Any,
        confluence_score: float,
        entry_price: float,
        atr: float,
    ) -> dict[str, Any]:
        try:
            rsi = float(row.get("m5_rsi_14", row.get("h1_rsi_14", 50.0)))
        except Exception:
            rsi = 50.0
        try:
            momentum = float(row.get("m5_macd_hist_slope", row.get("m1_momentum_3", 0.0)))
        except Exception:
            momentum = 0.0
        liquidity_sweep = bool(
            int(float(row.get("m5_liquidity_sweep", 0) or 0)) == 1
            or int(float(row.get("m5_sweep", 0) or 0)) == 1
            or int(float(row.get("m5_pinbar_bull", 0) or 0)) == 1
            or int(float(row.get("m5_pinbar_bear", 0) or 0)) == 1
        )
        zone_size = max(atr * 0.35, 1e-6)
        zone_bucket = int(entry_price / zone_size) if entry_price > 0 else 0
        return {
            "rsi": rsi,
            "momentum": momentum,
            "liquidity_sweep": liquidity_sweep,
            "confluence_score": float(confluence_score),
            "zone_bucket": zone_bucket,
        }

    def _enforce_symbol_cap(self, symbol: str) -> None:
        normalized = str(symbol).upper()
        live = [
            idea
            for idea in self.ideas.values()
            if idea.symbol == normalized and idea.status != IDEA_ARCHIVED
        ]
        if len(live) < max(1, int(self.max_active_ideas_per_symbol)):
            return
        live.sort(key=lambda item: item.timestamp_created)
        while len(live) >= max(1, int(self.max_active_ideas_per_symbol)):
            oldest = live.pop(0)
            oldest.status = IDEA_ARCHIVED
            oldest.cooldown_until = None

    def _archive_if_expired(self, *, idea: TradeIdea, now: datetime) -> None:
        age_seconds = max(0.0, (now - idea.timestamp_created).total_seconds())
        if idea.evaluation_count >= max(1, int(self.max_rechecks_per_idea)):
            idea.status = IDEA_ARCHIVED
            idea.cooldown_until = None
            return
        if age_seconds >= max(60, int(self.archive_minutes) * 60):
            idea.status = IDEA_ARCHIVED
            idea.cooldown_until = None

    @staticmethod
    def _session_seconds(mapping: dict[str, int], session_name: str, default: int) -> int:
        key = str(session_name or "").upper()
        if key in mapping:
            return max(1, int(mapping[key]))
        if "DEFAULT" in mapping:
            return max(1, int(mapping["DEFAULT"]))
        return max(1, int(default))

    @staticmethod
    def _structure_changed(previous: dict[str, Any], current: dict[str, Any]) -> bool:
        if not previous:
            return False
        prev_rsi = float(previous.get("rsi", 50.0))
        curr_rsi = float(current.get("rsi", 50.0))
        prev_momentum = abs(float(previous.get("momentum", 0.0)))
        curr_momentum = abs(float(current.get("momentum", 0.0)))
        prev_confluence = float(previous.get("confluence_score", 0.0))
        curr_confluence = float(current.get("confluence_score", 0.0))
        prev_liquidity = bool(previous.get("liquidity_sweep", False))
        curr_liquidity = bool(current.get("liquidity_sweep", False))
        prev_zone = int(previous.get("zone_bucket", 0))
        curr_zone = int(current.get("zone_bucket", 0))

        rsi_reactivation = (45.0 <= prev_rsi <= 55.0) and (curr_rsi < 40.0 or curr_rsi > 60.0)
        momentum_reactivation = curr_momentum >= max(0.03, prev_momentum + 0.015)
        liquidity_reactivation = curr_liquidity and (not prev_liquidity)
        confluence_reactivation = curr_confluence >= (prev_confluence + 0.5)
        zone_reactivation = curr_zone != prev_zone

        return bool(
            rsi_reactivation
            or momentum_reactivation
            or liquidity_reactivation
            or confluence_reactivation
            or zone_reactivation
        )

    @staticmethod
    def _idea_id(*, symbol: str, setup_type: str, side: str, entry_price: float, atr: float) -> str:
        zone_size = max(atr * 0.35, 1e-6)
        zone_bucket = int(entry_price / zone_size) if entry_price > 0 else 0
        return deterministic_id(str(symbol).upper(), str(setup_type).upper(), str(side).upper(), zone_bucket)

    @staticmethod
    def _allows_fast_reactivation(*, symbol: str, setup_type: str) -> bool:
        normalized_symbol = str(symbol or "").upper()
        normalized_setup = str(setup_type or "").upper()
        return normalized_symbol == "XAUUSD" and normalized_setup.startswith("XAUUSD_M5_GRID_SCALPER")
