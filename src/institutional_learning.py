from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import sqlite3

from src.symbol_universe import normalize_symbol_key
from src.utils import clamp, ensure_parent, utc_now

UTC = timezone.utc


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _side_sign(side: str) -> float:
    side_key = str(side or "").upper()
    if side_key == "BUY":
        return 1.0
    if side_key == "SELL":
        return -1.0
    return 0.0


@dataclass(frozen=True)
class CandleMasterySnapshot:
    symbol: str = ""
    timeframe: str = "M5"
    direction: str = "neutral"
    mastery_score: float = 0.5
    alignment_score: float = 0.0
    body_efficiency: float = 0.0
    wick_rejection_score: float = 0.0
    exhaustion_score: float = 0.0
    absorption_score: float = 0.0
    continuation_score: float = 0.0
    reversal_score: float = 0.0
    fakeout_score: float = 0.0
    liquidity_sweep_score: float = 0.0
    fvg_score: float = 0.0
    bos_score: float = 0.0
    choch_score: float = 0.0
    order_block_score: float = 0.0
    volume_profile_score: float = 0.5
    vwap_deviation_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InstitutionalConfluenceSnapshot:
    symbol: str = ""
    side: str = ""
    score: float = 0.5
    grade: str = "C"
    aggression_multiplier: float = 1.0
    risk_throttle: float = 1.0
    reasons: tuple[str, ...] = ()
    candle: CandleMasterySnapshot = field(default_factory=CandleMasterySnapshot)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candle"] = self.candle.as_dict()
        return payload


@dataclass(frozen=True)
class LiveShadowGap:
    symbol: str
    strategy: str
    live_expectancy_r: float
    shadow_expectancy_r: float
    gap_score: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvolutionProposal:
    strategy: str
    action: str
    confidence: float
    reason: str
    params: dict[str, float] = field(default_factory=dict)
    requires_shadow_trades: int = 25
    requires_walk_forward: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelfEvolutionReview:
    review_id: str
    created_at: str
    horizon: str
    live_trades: int
    shadow_trades: int
    win_rate: float
    expectancy_r: float
    profit_factor: float
    loss_root_causes: dict[str, int]
    live_shadow_gaps: tuple[LiveShadowGap, ...]
    proposals: tuple[EvolutionProposal, ...]
    memory: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["live_shadow_gaps"] = [item.as_dict() for item in self.live_shadow_gaps]
        payload["proposals"] = [item.as_dict() for item in self.proposals]
        return payload


def build_candle_mastery_from_row(
    row: Any,
    *,
    prefix: str = "m5",
    side: str = "",
    symbol: str = "",
) -> CandleMasterySnapshot:
    body_efficiency = clamp(abs(_float(row.get(f"{prefix}_body_efficiency"), 0.0)), 0.0, 1.0)
    upper_wick = clamp(_float(row.get(f"{prefix}_upper_wick_ratio"), 0.0), 0.0, 1.0)
    lower_wick = clamp(_float(row.get(f"{prefix}_lower_wick_ratio"), 0.0), 0.0, 1.0)
    trend_efficiency = clamp(_float(row.get(f"{prefix}_trend_efficiency_16"), 0.0), 0.0, 1.0)
    volume_ratio = clamp(_float(row.get(f"{prefix}_volume_ratio_20"), 1.0) / 2.5, 0.0, 1.0)
    wick_rejection = max(upper_wick, lower_wick)
    candle_direction = _float(row.get(f"{prefix}_candle_direction"), _float(row.get(f"{prefix}_bullish"), 0.0) - _float(row.get(f"{prefix}_bearish"), 0.0))
    liquidity_sweep = clamp(_float(row.get(f"{prefix}_liquidity_sweep"), 0.0), 0.0, 1.0)
    fvg = clamp(max(_float(row.get(f"{prefix}_fvg_bull"), 0.0), _float(row.get(f"{prefix}_fvg_bear"), 0.0)), 0.0, 1.0)
    bos = clamp(max(_float(row.get(f"{prefix}_bos_bull"), 0.0), _float(row.get(f"{prefix}_bos_bear"), 0.0)), 0.0, 1.0)
    choch = clamp(max(_float(row.get(f"{prefix}_choch_bull"), 0.0), _float(row.get(f"{prefix}_choch_bear"), 0.0)), 0.0, 1.0)
    fakeout = clamp(max(_float(row.get(f"{prefix}_fakeout_high"), 0.0), _float(row.get(f"{prefix}_fakeout_low"), 0.0)), 0.0, 1.0)
    absorption = clamp(_float(row.get(f"{prefix}_absorption_proxy"), 0.0), 0.0, 1.0)
    order_block = clamp(max(_float(row.get(f"{prefix}_order_block_bull"), 0.0), _float(row.get(f"{prefix}_order_block_bear"), 0.0)), 0.0, 1.0)
    volume_profile = clamp(_float(row.get(f"{prefix}_volume_profile_pressure"), 0.5), 0.0, 1.0)
    vwap_deviation = clamp(abs(_float(row.get(f"{prefix}_vwap_deviation_atr"), 0.0)) / 2.0, 0.0, 1.0)
    exhaustion = clamp(
        (wick_rejection * 0.45)
        + (fakeout * 0.20)
        + (absorption * 0.20)
        + (max(0.0, 1.0 - body_efficiency) * 0.15),
        0.0,
        1.0,
    )
    continuation = clamp((body_efficiency * 0.34) + (trend_efficiency * 0.28) + (bos * 0.20) + (fvg * 0.10) + (volume_ratio * 0.08), 0.0, 1.0)
    reversal = clamp((liquidity_sweep * 0.32) + (choch * 0.25) + (wick_rejection * 0.20) + (absorption * 0.15) + (fakeout * 0.08), 0.0, 1.0)
    direction = "bullish" if candle_direction > 0.10 else "bearish" if candle_direction < -0.10 else "neutral"
    directional_alignment = clamp(candle_direction * _side_sign(side), -1.0, 1.0) if side else 0.0
    mastery = clamp(
        (continuation * 0.30)
        + (reversal * 0.24)
        + (volume_profile * 0.14)
        + ((1.0 - vwap_deviation) * 0.12)
        + (order_block * 0.10)
        + ((1.0 - exhaustion) * 0.10),
        0.0,
        1.0,
    )
    return CandleMasterySnapshot(
        symbol=normalize_symbol_key(symbol),
        timeframe=prefix.upper(),
        direction=direction,
        mastery_score=float(mastery),
        alignment_score=float(directional_alignment),
        body_efficiency=float(body_efficiency),
        wick_rejection_score=float(wick_rejection),
        exhaustion_score=float(exhaustion),
        absorption_score=float(absorption),
        continuation_score=float(continuation),
        reversal_score=float(reversal),
        fakeout_score=float(fakeout),
        liquidity_sweep_score=float(liquidity_sweep),
        fvg_score=float(fvg),
        bos_score=float(bos),
        choch_score=float(choch),
        order_block_score=float(order_block),
        volume_profile_score=float(volume_profile),
        vwap_deviation_score=float(vwap_deviation),
    )


def build_institutional_confluence(
    row: Any,
    *,
    symbol: str = "",
    side: str = "",
) -> InstitutionalConfluenceSnapshot:
    candle = build_candle_mastery_from_row(row, prefix="m5", side=side, symbol=symbol)
    multi_tf = clamp(_float(row.get("multi_tf_alignment_score"), 0.5), 0.0, 1.0)
    fractal = clamp(_float(row.get("fractal_persistence_score"), 0.5), 0.0, 1.0)
    compression = clamp(_float(row.get("compression_expansion_score"), 0.0), 0.0, 1.0)
    session = clamp(_float(row.get("session_aggression_score"), 0.5), 0.0, 1.0)
    instability = clamp(_float(row.get("market_instability_score"), 0.0), 0.0, 1.0)
    execution = clamp(_float(row.get("execution_edge_score"), 0.65), 0.0, 1.0)
    live_shadow_gap = clamp(_float(row.get("live_shadow_gap_risk_score"), 0.0), 0.0, 1.0)
    score = clamp(
        (candle.mastery_score * 0.30)
        + (multi_tf * 0.22)
        + (fractal * 0.14)
        + (compression * 0.10)
        + (session * 0.10)
        + (execution * 0.09)
        - (instability * 0.10)
        - (live_shadow_gap * 0.12),
        0.0,
        1.0,
    )
    grade = "A+" if score >= 0.82 else "A" if score >= 0.72 else "B" if score >= 0.62 else "C" if score >= 0.50 else "D"
    aggression = adaptive_aggression_multiplier(
        confluence_score=score,
        execution_edge=execution,
        live_shadow_gap=live_shadow_gap,
        market_instability=instability,
    )
    reasons = []
    if candle.liquidity_sweep_score >= 0.5:
        reasons.append("liquidity_sweep")
    if candle.fvg_score >= 0.5:
        reasons.append("fvg_displacement")
    if candle.bos_score >= 0.5:
        reasons.append("bos")
    if candle.choch_score >= 0.5:
        reasons.append("choch")
    if live_shadow_gap >= 0.35:
        reasons.append("live_shadow_gap_throttle")
    if instability >= 0.55:
        reasons.append("instability_throttle")
    return InstitutionalConfluenceSnapshot(
        symbol=normalize_symbol_key(symbol),
        side=str(side or "").upper(),
        score=float(score),
        grade=grade,
        aggression_multiplier=float(aggression),
        risk_throttle=float(clamp(1.0 - max(instability, live_shadow_gap) * 0.55, 0.35, 1.0)),
        reasons=tuple(reasons),
        candle=candle,
    )


def adaptive_aggression_multiplier(
    *,
    confluence_score: float,
    execution_edge: float,
    live_shadow_gap: float = 0.0,
    market_instability: float = 0.0,
    equity_curve_heat: float = 0.0,
) -> float:
    raw = 0.75 + (clamp(confluence_score, 0.0, 1.0) * 0.42) + (clamp(execution_edge, 0.0, 1.0) * 0.20)
    raw -= clamp(live_shadow_gap, 0.0, 1.0) * 0.28
    raw -= clamp(market_instability, 0.0, 1.0) * 0.22
    raw -= clamp(equity_curve_heat, 0.0, 1.0) * 0.18
    return float(clamp(raw, 0.55, 1.35))


class SelfEvolutionEngine:
    def __init__(self, db_path: Path, logger: Any | None = None) -> None:
        self.db_path = db_path
        self.logger = logger
        ensure_parent(db_path)
        self._init_db()

    def review(
        self,
        *,
        live_trades: Iterable[dict[str, Any]],
        shadow_trades: Iterable[dict[str, Any]] = (),
        horizon: str = "4h",
        now_utc: datetime | None = None,
    ) -> SelfEvolutionReview:
        live = [dict(item) for item in live_trades]
        shadow = [dict(item) for item in shadow_trades]
        now = (now_utc or utc_now()).astimezone(UTC)
        pnl = [_float(item.get("pnl_r"), 0.0) for item in live]
        wins = [item for item in pnl if item > 0.0]
        losses = [abs(item) for item in pnl if item < 0.0]
        win_rate = float(len(wins) / len(pnl)) if pnl else 0.0
        expectancy = float(sum(pnl) / len(pnl)) if pnl else 0.0
        gross_profit = float(sum(wins))
        gross_loss = float(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (99.0 if gross_profit > 0.0 else 0.0)
        root_causes = self._loss_root_causes(live)
        gaps = tuple(self._live_shadow_gaps(live, shadow))
        proposals = tuple(self._build_proposals(root_causes=root_causes, gaps=gaps, expectancy=expectancy, win_rate=win_rate))
        memory = {
            "strong_sessions": self._top_bucket(live, "session_name"),
            "strong_regimes": self._top_bucket(live, "regime_state"),
            "weak_causes": root_causes,
            "review_note": "proposals_require_shadow_and_walk_forward_gate",
        }
        review_seed = json.dumps(
            {
                "created_at": now.isoformat(),
                "horizon": str(horizon),
                "live": len(live),
                "shadow": len(shadow),
                "expectancy": round(expectancy, 6),
                "win_rate": round(win_rate, 6),
            },
            sort_keys=True,
        )
        review_digest = hashlib.sha256(review_seed.encode("utf-8")).hexdigest()[:10]
        review = SelfEvolutionReview(
            review_id=f"evo-{now.strftime('%Y%m%dT%H%M%S%f')}-{review_digest}",
            created_at=now.isoformat(),
            horizon=str(horizon),
            live_trades=len(live),
            shadow_trades=len(shadow),
            win_rate=win_rate,
            expectancy_r=expectancy,
            profit_factor=float(profit_factor),
            loss_root_causes=root_causes,
            live_shadow_gaps=gaps,
            proposals=proposals,
            memory=memory,
        )
        self.record_review(review)
        return review

    def compressed_memory(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect(row_factory=True) as connection:
            rows = connection.execute(
                """
                SELECT created_at, horizon, win_rate, expectancy_r, profit_factor, memory_json
                FROM institutional_evolution_reviews
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [
            {
                "created_at": str(row["created_at"]),
                "horizon": str(row["horizon"]),
                "win_rate": float(row["win_rate"] or 0.0),
                "expectancy_r": float(row["expectancy_r"] or 0.0),
                "profit_factor": float(row["profit_factor"] or 0.0),
                "memory": json.loads(str(row["memory_json"] or "{}")),
            }
            for row in rows
        ]

    def record_review(self, review: SelfEvolutionReview) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO institutional_evolution_reviews (
                    review_id, created_at, horizon, live_trades, shadow_trades,
                    win_rate, expectancy_r, profit_factor, root_causes_json,
                    gaps_json, proposals_json, memory_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review.review_id,
                    review.created_at,
                    review.horizon,
                    review.live_trades,
                    review.shadow_trades,
                    review.win_rate,
                    review.expectancy_r,
                    review.profit_factor,
                    json.dumps(review.loss_root_causes, sort_keys=True),
                    json.dumps([item.as_dict() for item in review.live_shadow_gaps], sort_keys=True),
                    json.dumps([item.as_dict() for item in review.proposals], sort_keys=True),
                    json.dumps(review.memory, sort_keys=True),
                ),
            )
            connection.commit()

    @staticmethod
    def _loss_root_causes(trades: list[dict[str, Any]]) -> dict[str, int]:
        causes: dict[str, int] = {}
        for trade in trades:
            if _float(trade.get("pnl_r"), 0.0) >= 0.0:
                continue
            if _float(trade.get("institutional_confluence_score"), 0.5) < 0.55:
                causes["weak_confluence"] = causes.get("weak_confluence", 0) + 1
            if _float(trade.get("candle_mastery_score"), 0.5) < 0.52:
                causes["poor_candle_structure"] = causes.get("poor_candle_structure", 0) + 1
            if _float(trade.get("execution_quality_fit"), 0.7) < 0.55 or _float(trade.get("slippage_points"), 0.0) > 0.0:
                causes["execution_drag"] = causes.get("execution_drag", 0) + 1
            if _float(trade.get("live_shadow_gap_score"), 0.0) > 0.35:
                causes["live_shadow_gap"] = causes.get("live_shadow_gap", 0) + 1
            if str(trade.get("news_state") or "").lower().find("block") >= 0:
                causes["news_distortion"] = causes.get("news_distortion", 0) + 1
        return causes

    @staticmethod
    def _live_shadow_gaps(live: list[dict[str, Any]], shadow: list[dict[str, Any]]) -> list[LiveShadowGap]:
        def grouped(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[float]]:
            output: dict[tuple[str, str], list[float]] = {}
            for row in rows:
                key = (normalize_symbol_key(row.get("symbol")), str(row.get("strategy_key") or row.get("setup") or "UNKNOWN").upper())
                output.setdefault(key, []).append(_float(row.get("pnl_r"), 0.0))
            return output

        live_group = grouped(live)
        shadow_group = grouped(shadow)
        gaps: list[LiveShadowGap] = []
        for key, live_values in live_group.items():
            shadow_values = shadow_group.get(key)
            if not shadow_values:
                continue
            live_exp = sum(live_values) / max(1, len(live_values))
            shadow_exp = sum(shadow_values) / max(1, len(shadow_values))
            gap = clamp(abs(shadow_exp - live_exp) / 1.5, 0.0, 1.0)
            reason = "shadow_outperforms_live" if shadow_exp > live_exp else "live_outperforms_shadow"
            if gap >= 0.18:
                gaps.append(LiveShadowGap(key[0], key[1], float(live_exp), float(shadow_exp), float(gap), reason))
        return gaps

    @staticmethod
    def _build_proposals(
        *,
        root_causes: dict[str, int],
        gaps: tuple[LiveShadowGap, ...],
        expectancy: float,
        win_rate: float,
    ) -> list[EvolutionProposal]:
        proposals: list[EvolutionProposal] = []
        if root_causes.get("poor_candle_structure", 0) >= 2:
            proposals.append(
                EvolutionProposal(
                    strategy="GLOBAL",
                    action="tighten_candle_mastery_filter",
                    confidence=0.68,
                    reason="multiple_losses_from_weak_candle_structure",
                    params={"min_candle_mastery_score": 0.55},
                )
            )
        if root_causes.get("execution_drag", 0) >= 2:
            proposals.append(
                EvolutionProposal(
                    strategy="GLOBAL",
                    action="reduce_size_in_rough_execution_minutes",
                    confidence=0.72,
                    reason="execution_drag_detected",
                    params={"execution_edge_min": 0.58},
                )
            )
        for gap in gaps:
            proposals.append(
                EvolutionProposal(
                    strategy=gap.strategy,
                    action="shadow_gap_investigation",
                    confidence=clamp(0.50 + gap.gap_score * 0.40, 0.0, 1.0),
                    reason=gap.reason,
                    params={"live_shadow_gap_score": gap.gap_score},
                    requires_shadow_trades=40,
                )
            )
        if expectancy < 0.0 and win_rate < 0.48:
            proposals.append(
                EvolutionProposal(
                    strategy="GLOBAL",
                    action="quarantine_until_walk_forward_recovers",
                    confidence=0.80,
                    reason="negative_expectancy_and_low_win_rate",
                    params={"risk_throttle": 0.50},
                    requires_shadow_trades=50,
                )
            )
        return proposals

    @staticmethod
    def _top_bucket(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        buckets: dict[str, list[float]] = {}
        for row in rows:
            bucket = str(row.get(key) or "UNKNOWN").upper()
            buckets.setdefault(bucket, []).append(_float(row.get("pnl_r"), 0.0))
        ranked = sorted(
            (
                {"name": name, "trades": len(values), "expectancy_r": sum(values) / max(1, len(values))}
                for name, values in buckets.items()
            ),
            key=lambda item: (float(item["expectancy_r"]), int(item["trades"])),
            reverse=True,
        )
        return ranked[:5]

    def _connect(self, row_factory: bool = False) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        if row_factory:
            connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS institutional_evolution_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    live_trades INTEGER NOT NULL,
                    shadow_trades INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    expectancy_r REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    root_causes_json TEXT NOT NULL,
                    gaps_json TEXT NOT NULL,
                    proposals_json TEXT NOT NULL,
                    memory_json TEXT NOT NULL
                )
                """
            )
            connection.commit()
