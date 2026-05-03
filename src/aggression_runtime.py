from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.symbol_universe import normalize_symbol_key as _canonical_symbol_key, symbol_asset_class
from src.utils import clamp


UTC = timezone.utc


def normalize_symbol_key(value: str) -> str:
    return _canonical_symbol_key(value)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _direction_sign(side: str) -> float:
    return 1.0 if str(side or "").upper() == "BUY" else -1.0 if str(side or "").upper() == "SELL" else 1.0


def _currency_pair(symbol_key: str) -> tuple[str, str]:
    if len(symbol_key) == 6 and symbol_asset_class(symbol_key) == "forex":
        return symbol_key[:3], symbol_key[3:6]
    return "", ""


def _currency_macro_score(
    currency: str,
    *,
    dxy_value: float,
    yields_value: float,
    nas100_value: float,
    usoil_value: float,
    usd_liquidity_value: float,
    risk_sentiment_value: float,
) -> float:
    normalized = str(currency or "").upper()
    if normalized == "USD":
        return clamp((dxy_value * 0.55) + (yields_value * 0.25) - (risk_sentiment_value * 0.15), -1.0, 1.0)
    if normalized == "JPY":
        return clamp((-risk_sentiment_value * 0.52) - (yields_value * 0.28) - (usd_liquidity_value * 0.10), -1.0, 1.0)
    if normalized in {"EUR", "GBP"}:
        return clamp((-dxy_value * 0.48) + (risk_sentiment_value * 0.16) + (usd_liquidity_value * 0.16), -1.0, 1.0)
    if normalized in {"AUD", "NZD"}:
        return clamp((risk_sentiment_value * 0.42) + (nas100_value * 0.18) + (usoil_value * 0.08) - (dxy_value * 0.18), -1.0, 1.0)
    if normalized == "CAD":
        return clamp((usoil_value * 0.40) + (risk_sentiment_value * 0.12) - (dxy_value * 0.18), -1.0, 1.0)
    if normalized == "CHF":
        return clamp((-risk_sentiment_value * 0.34) + (dxy_value * 0.10), -1.0, 1.0)
    return 0.0


def _default_lead_lag_weights(symbol_key: str, asset_class: str) -> dict[str, float]:
    base, quote = _currency_pair(symbol_key)
    if symbol_key == "XAUUSD":
        return {"dxy": 0.40, "yields": 0.25, "nas100": 0.20, "usoil": 0.15}
    if symbol_key == "XAGUSD":
        return {"dxy": 0.28, "yields": 0.14, "risk_sentiment": 0.28, "nas100": 0.18, "usoil": 0.12}
    if symbol_key == "BTCUSD":
        return {"nas100": 0.34, "dxy": 0.22, "usd_liquidity": 0.20, "weekend_volatility": 0.14, "risk_sentiment": 0.10}
    if asset_class == "crypto":
        return {"btc": 0.30, "nas100": 0.24, "dxy": 0.18, "usd_liquidity": 0.16, "weekend_volatility": 0.12}
    if symbol_key == "NAS100":
        return {"yields": 0.32, "usd_liquidity": 0.26, "dxy": 0.18, "risk_sentiment": 0.14, "btc": 0.10}
    if symbol_key == "USOIL":
        return {"dxy": 0.30, "risk_sentiment": 0.22, "nas100": 0.18, "usd_liquidity": 0.16, "yields": 0.14}
    if asset_class == "equity":
        return {"nas100": 0.36, "yields": 0.24, "usd_liquidity": 0.18, "risk_sentiment": 0.14, "dxy": 0.08}
    if quote == "JPY" or base == "JPY":
        return {"fx_spread": 0.42, "risk_sentiment": 0.26, "yields": 0.20, "nas100": 0.12}
    if asset_class == "forex" and "USD" in {base, quote}:
        return {"fx_spread": 0.56, "dxy": 0.22, "yields": 0.12, "risk_sentiment": 0.10}
    if asset_class == "forex":
        return {"fx_spread": 0.58, "risk_sentiment": 0.24, "usoil": 0.10, "dxy": 0.08}
    if asset_class == "index":
        return {"nas100": 0.36, "risk_sentiment": 0.24, "usd_liquidity": 0.18, "yields": 0.14, "dxy": 0.08}
    if asset_class == "commodity":
        return {"dxy": 0.28, "risk_sentiment": 0.24, "usoil": 0.18, "nas100": 0.18, "yields": 0.12}
    return {}


@dataclass(frozen=True)
class MicrostructureScore:
    ready: bool = False
    symbol: str = ""
    direction: str = "neutral"
    confidence: float = 0.0
    composite_score: float = 0.5
    directional_bias: float = 0.0
    alignment_score: float = 0.0
    sweep_velocity: float = 0.0
    absorption_score: float = 0.0
    iceberg_score: float = 0.0
    dom_imbalance: float = 0.0
    spread_shock_score: float = 0.0
    quote_pull_stack_score: float = 0.0
    pressure_score: float = 0.5
    cumulative_delta_score: float = 0.0
    depth_imbalance: float = 0.0
    drift_score: float = 0.0
    spread_stability: float = 0.5
    tick_count: int = 0
    book_levels: int = 0
    stale: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeadLagSnapshot:
    symbol: str = ""
    direction: str = "neutral"
    confidence: float = 0.0
    alignment_score: float = 0.0
    agreement_score: float = 0.5
    disagreement_penalty: float = 0.0
    weights_used: dict[str, float] = field(default_factory=dict)
    components: dict[str, float] = field(default_factory=dict)
    details: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventDirective:
    symbol: str = ""
    base_class: str = "general_macro"
    sub_class: str = "general_macro"
    playbook: str = "fade"
    risk_bias: str = "neutral"
    confidence: float = 0.0
    scheduled: bool = False
    pre_position_allowed: bool = False
    pre_position_window_minutes: int = 0
    wait_minutes_after: int = 0
    affected_symbols: tuple[str, ...] = ()
    reason: str = ""
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowVariantScore:
    variant_id: str
    promotion_score: float
    expectancy_r: float
    profit_factor: float
    slippage_adjusted_score: float
    promoted: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlockedOpportunityRecord:
    symbol: str
    blocker: str
    setup: str = ""
    would_have_won: bool = False
    score: float = 0.0
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionMinuteProfile:
    minute_of_day: int
    session_name: str
    quality_score: float
    size_multiplier: float
    state: str
    spread_score: float
    slippage_score: float
    fill_score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_microstructure_score(
    snapshot: dict[str, Any] | None,
    *,
    symbol: str = "",
    side: str = "",
) -> MicrostructureScore:
    payload = dict(snapshot or {})
    ready = bool(payload.get("ready", False))
    pressure_score = clamp(_float(payload.get("pressure_score"), 0.5), 0.0, 1.0)
    cumulative_delta_score = clamp(_float(payload.get("cumulative_delta_score"), 0.0), -1.0, 1.0)
    depth_imbalance = clamp(_float(payload.get("depth_imbalance", payload.get("dom_imbalance")), 0.0), -1.0, 1.0)
    drift_score = clamp(_float(payload.get("drift_score"), 0.0), -1.0, 1.0)
    spread_stability = clamp(_float(payload.get("spread_stability"), 0.5), 0.0, 1.0)
    dom_imbalance = clamp(_float(payload.get("dom_imbalance", depth_imbalance), depth_imbalance), -1.0, 1.0)
    sweep_velocity = clamp(
        _float(
            payload.get(
                "sweep_velocity",
                min(1.0, (abs(cumulative_delta_score) * 0.70) + (abs(drift_score) * 0.45)),
            )
        ),
        0.0,
        1.0,
    )
    absorption_score = clamp(
        _float(
            payload.get(
                "absorption_score",
                max(0.0, min(1.0, spread_stability - abs(drift_score - cumulative_delta_score))),
            )
        ),
        0.0,
        1.0,
    )
    iceberg_score = clamp(
        _float(
            payload.get(
                "iceberg_score",
                payload.get("refill_score", max(0.0, (abs(dom_imbalance) * 0.40) + (absorption_score * 0.60))),
            )
        ),
        0.0,
        1.0,
    )
    spread_shock_score = clamp(
        _float(
            payload.get(
                "spread_shock_score",
                max(0.0, (0.55 - spread_stability) * 2.0),
            )
        ),
        0.0,
        1.0,
    )
    quote_pull_stack_score = clamp(
        _float(
            payload.get(
                "quote_pull_stack_score",
                max(0.0, (abs(dom_imbalance) * 0.50) + (sweep_velocity * 0.35) - (spread_shock_score * 0.20)),
            )
        ),
        0.0,
        1.0,
    )
    directional_bias = clamp(
        ((pressure_score - 0.5) * 2.0 * 0.40)
        + (cumulative_delta_score * 0.24)
        + (dom_imbalance * 0.18)
        + (drift_score * 0.12)
        + ((0.5 - spread_shock_score) * 0.06),
        -1.0,
        1.0,
    )
    confidence = clamp(
        _float(payload.get("confidence"), 0.0)
        or (
            (abs(directional_bias) * 0.64)
            + (sweep_velocity * 0.14)
            + (absorption_score * 0.08)
            + (iceberg_score * 0.08)
            + (quote_pull_stack_score * 0.08)
            - (spread_shock_score * 0.10)
        ),
        0.0,
        1.0,
    )
    composite_score = clamp(
        0.5 + (directional_bias * 0.42) + ((confidence - 0.5) * 0.16),
        0.0,
        1.0,
    )
    direction = "bullish" if directional_bias >= 0.15 else "bearish" if directional_bias <= -0.15 else "neutral"
    alignment_score = clamp(directional_bias * _direction_sign(side), -1.0, 1.0)
    tick_count = max(0, int(_float(payload.get("tick_count"), 0.0)))
    book_levels = max(0, int(_float(payload.get("book_levels"), 0.0)))
    stale = bool(not ready or tick_count < 8 or book_levels < 1)
    return MicrostructureScore(
        ready=bool(ready),
        symbol=normalize_symbol_key(symbol),
        direction=direction,
        confidence=float(confidence),
        composite_score=float(composite_score),
        directional_bias=float(directional_bias),
        alignment_score=float(alignment_score),
        sweep_velocity=float(sweep_velocity),
        absorption_score=float(absorption_score),
        iceberg_score=float(iceberg_score),
        dom_imbalance=float(dom_imbalance),
        spread_shock_score=float(spread_shock_score),
        quote_pull_stack_score=float(quote_pull_stack_score),
        pressure_score=float(pressure_score),
        cumulative_delta_score=float(cumulative_delta_score),
        depth_imbalance=float(depth_imbalance),
        drift_score=float(drift_score),
        spread_stability=float(spread_stability),
        tick_count=int(tick_count),
        book_levels=int(book_levels),
        stale=bool(stale),
    )


def build_lead_lag_snapshot(
    *,
    symbol: str,
    side: str = "",
    context: dict[str, Any] | None = None,
    weights_config: dict[str, dict[str, float]] | None = None,
) -> LeadLagSnapshot:
    payload = dict(context or {})
    symbol_key = normalize_symbol_key(symbol)
    asset_class = symbol_asset_class(symbol_key)
    weights = dict((weights_config or {}).get(symbol_key) or _default_lead_lag_weights(symbol_key, asset_class))
    if not weights:
        return LeadLagSnapshot(symbol=symbol_key)

    dxy_value = _float(payload.get("dxy_ret_1"), _float(payload.get("dxy_ret_5"), 0.0))
    yields_value = _float(payload.get("us10y_ret_5"), _float(payload.get("yield_proxy_ret_5"), 0.0))
    nas100_value = _float(payload.get("nas100_ret_5"), _float(payload.get("nas_ret_5"), 0.0))
    usoil_value = _float(payload.get("usoil_ret_5"), _float(payload.get("oil_ret_5"), 0.0))
    btc_value = _float(payload.get("btc_ret_5"), _float(payload.get("btc_ret_1"), 0.0))
    usd_liquidity_value = _float(
        payload.get("usd_liquidity_score"),
        _float(payload.get("eurusd_ret_5"), 0.0) - _float(payload.get("usdjpy_ret_5"), 0.0),
    )
    weekend_volatility_value = _float(
        payload.get("weekend_volatility_score"),
        _float(payload.get("btc_weekend_gap_score"), _float(payload.get("weekend_gap_score"), 0.0)),
    )
    risk_sentiment_value = _float(
        payload.get("risk_sentiment_score"),
        clamp((nas100_value * 0.62) - (dxy_value * 0.22) - (max(0.0, yields_value) * 0.16), -1.0, 1.0),
    )
    base_currency, quote_currency = _currency_pair(symbol_key)
    fx_spread_value = 0.0
    if base_currency and quote_currency:
        fx_spread_value = clamp(
            _currency_macro_score(
                base_currency,
                dxy_value=dxy_value,
                yields_value=yields_value,
                nas100_value=nas100_value,
                usoil_value=usoil_value,
                usd_liquidity_value=usd_liquidity_value,
                risk_sentiment_value=risk_sentiment_value,
            )
            - _currency_macro_score(
                quote_currency,
                dxy_value=dxy_value,
                yields_value=yields_value,
                nas100_value=nas100_value,
                usoil_value=usoil_value,
                usd_liquidity_value=usd_liquidity_value,
                risk_sentiment_value=risk_sentiment_value,
            ),
            -1.0,
            1.0,
        )
    components = {
        "dxy": dxy_value,
        "yields": yields_value,
        "nas100": nas100_value,
        "usoil": usoil_value,
        "btc": btc_value,
        "usd_liquidity": usd_liquidity_value,
        "weekend_volatility": weekend_volatility_value,
        "risk_sentiment": risk_sentiment_value,
        "fx_spread": fx_spread_value,
    }
    if symbol_key in {"XAUUSD", "BTCUSD", "DOGUSD", "TRUMPUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "USOIL", "NAS100", "AAPL", "NVIDIA"}:
        components["dxy"] *= -1.0
    if symbol_key in {"XAUUSD", "XAGUSD", "NAS100", "AAPL", "NVIDIA", "USOIL"}:
        components["yields"] *= -1.0
    if symbol_key == "XAUUSD":
        components["nas100"] *= -1.0
    if symbol_key in {"DOGUSD", "TRUMPUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD"}:
        components["btc"] = clamp((btc_value * 0.75) + (nas100_value * 0.25), -1.0, 1.0)
    if asset_class == "equity":
        components["risk_sentiment"] = clamp((risk_sentiment_value * 0.65) + (nas100_value * 0.35), -1.0, 1.0)
    details: list[str] = []
    agreement = 0.0
    disagreement = 0.0
    weighted_sum = 0.0
    for name, weight in weights.items():
        value = clamp(_float(components.get(name), 0.0), -1.0, 1.0)
        weighted_value = value * float(weight)
        weighted_sum += weighted_value
        if value >= 0.0:
            agreement += abs(weighted_value)
            if value > 0.05:
                details.append(f"{name}_supportive")
        else:
            disagreement += abs(weighted_value)
            if value < -0.05:
                details.append(f"{name}_contrary")
    alignment_score = clamp(weighted_sum, -1.0, 1.0)
    side_adjusted = clamp(alignment_score * _direction_sign(side), -1.0, 1.0)
    confidence = clamp((abs(alignment_score) * 0.78) + (agreement * 0.18), 0.0, 1.0)
    disagreement_penalty = clamp(disagreement, 0.0, 1.0)
    direction = "bullish" if alignment_score >= 0.12 else "bearish" if alignment_score <= -0.12 else "neutral"
    return LeadLagSnapshot(
        symbol=symbol_key,
        direction=direction,
        confidence=float(confidence),
        alignment_score=float(side_adjusted),
        agreement_score=float(clamp(agreement, 0.0, 1.0)),
        disagreement_penalty=float(disagreement_penalty),
        weights_used={str(key): float(value) for key, value in weights.items()},
        components={str(key): float(components.get(key, 0.0)) for key in weights.keys()},
        details=details,
    )


def build_event_directive(
    *,
    symbol: str,
    news_snapshot: dict[str, Any] | None,
    lead_lag: LeadLagSnapshot | None = None,
    microstructure: MicrostructureScore | None = None,
    playbook_map: dict[str, str] | None = None,
) -> EventDirective:
    payload = dict(news_snapshot or {})
    symbol_key = normalize_symbol_key(symbol)
    asset_class = symbol_asset_class(symbol_key)
    primary_category = str(payload.get("news_primary_category") or "general_macro").strip().lower() or "general_macro"
    next_event = dict(payload.get("next_macro_event") or {})
    next_title = str(next_event.get("title") or "").strip()
    news_direction = str(payload.get("news_bias_direction") or "neutral").strip().lower()
    scheduled = bool(next_event) or bool(payload.get("event_risk_window_active"))
    default_playbooks = {
        "central_bank": "wait_then_retest",
        "inflation": "wait_then_retest",
        "labor": "wait_then_retest",
        "growth": "wait_then_retest",
        "monetary_policy": "wait_then_retest",
        "risk_sentiment": "risk_on_follow",
        "geopolitical": "fade",
        "commodity": "breakout",
        "inventory": "breakout",
        "commodity_supply": "breakout",
        "crypto": "swing_hold",
        "crypto_regulatory": "swing_hold",
        "regulatory": "swing_hold",
        "equity_earnings": "breakout",
        "technology": "breakout",
        "general_macro": "fade",
    }
    playbook_lookup = {**default_playbooks, **dict(playbook_map or {})}
    base_class = primary_category
    title_upper = next_title.upper()
    sub_class = primary_category
    if any(token in title_upper for token in ("FOMC", "FED", "ECB", "BOJ", "RBA", "RBNZ", "RATE DECISION")):
        base_class = "central_bank"
        sub_class = "rate_decision"
    elif "CPI" in title_upper or "PCE" in title_upper or "INFLATION" in title_upper:
        base_class = "inflation"
        sub_class = "price_pressure"
    elif any(token in title_upper for token in ("NFP", "EMPLOYMENT", "JOBS", "PAYROLLS", "UNEMPLOYMENT")):
        base_class = "labor"
        sub_class = "labor"
    elif any(token in title_upper for token in ("GDP", "PMI", "RETAIL SALES")):
        base_class = "growth"
        sub_class = "growth"
    elif any(token in title_upper for token in ("EIA", "CRUDE", "INVENTORY", "OPEC")):
        base_class = "inventory"
        sub_class = "commodity_supply"
    elif any(token in title_upper for token in ("SEC", "ETF", "CRYPTO", "BITCOIN", "STABLECOIN")):
        base_class = "crypto_regulatory"
        sub_class = "crypto_regulatory"
    elif any(token in title_upper for token in ("EARNINGS", "GUIDANCE", "REVENUE", "EPS", "NVDA", "NVIDIA", "AAPL", "APPLE", "MAGNIFICENT 7", "AI CHIP")):
        base_class = "equity_earnings"
        sub_class = "equity_earnings"
    playbook = str(playbook_lookup.get(sub_class) or playbook_lookup.get(base_class) or "fade")
    confidence = clamp(_float(payload.get("news_confidence"), 0.0), 0.0, 1.0)
    if lead_lag is not None:
        confidence = clamp(confidence + (float(lead_lag.confidence) * 0.12), 0.0, 1.0)
    if microstructure is not None:
        confidence = clamp(confidence + (float(microstructure.confidence) * 0.10), 0.0, 1.0)
    affected_symbols: tuple[str, ...]
    if base_class in {"crypto", "crypto_regulatory"}:
        affected_symbols = ("BTCUSD", "ETHUSD", "SOLUSD", "DOGUSD", "TRUMPUSD", "NAS100")
    elif base_class in {"commodity", "inventory", "commodity_supply"}:
        affected_symbols = ("USOIL", "XAUUSD", "XAGUSD")
    elif base_class in {"equity_earnings", "technology"}:
        affected_symbols = tuple(dict.fromkeys((symbol_key, "NAS100", "AAPL", "NVIDIA")))
    elif asset_class == "equity":
        affected_symbols = tuple(dict.fromkeys((symbol_key, "NAS100", "USDJPY")))
    elif asset_class == "index":
        affected_symbols = tuple(dict.fromkeys((symbol_key, "USDJPY", "AAPL", "NVIDIA")))
    elif asset_class == "crypto":
        affected_symbols = ("BTCUSD", "ETHUSD", "SOLUSD", "DOGUSD", "TRUMPUSD", "NAS100")
    elif asset_class == "commodity":
        affected_symbols = tuple(dict.fromkeys((symbol_key, "XAUUSD", "XAGUSD", "USOIL")))
    else:
        affected_symbols = ("XAUUSD", "XAGUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "NAS100", "AAPL", "NVIDIA")
    pre_position_allowed = bool(
        scheduled
        and playbook in {"wait_then_retest", "breakout", "risk_on_follow", "swing_hold"}
        and confidence >= 0.78
        and float((lead_lag or LeadLagSnapshot()).alignment_score) >= -0.05
        and float((microstructure or MicrostructureScore()).alignment_score) >= -0.05
    )
    wait_minutes_after = 3 if playbook == "wait_then_retest" else 0
    risk_bias = str(news_direction or "neutral")
    reason = next_title or str(payload.get("session_bias_summary") or "")
    return EventDirective(
        symbol=symbol_key,
        base_class=base_class,
        sub_class=sub_class,
        playbook=playbook,
        risk_bias=risk_bias,
        confidence=float(confidence),
        scheduled=bool(scheduled),
        pre_position_allowed=bool(pre_position_allowed),
        pre_position_window_minutes=45 if pre_position_allowed else 0,
        wait_minutes_after=int(wait_minutes_after),
        affected_symbols=affected_symbols,
        reason=reason,
        source=str(payload.get("news_source_used") or ""),
    )


def score_shadow_variant(
    variant: dict[str, Any] | None,
    *,
    performance: dict[str, Any] | None = None,
    promotion_threshold: float = 0.64,
) -> ShadowVariantScore:
    payload = dict(variant or {})
    perf = dict(performance or {})
    score_hint = clamp(_float(payload.get("score_hint"), 0.5), 0.0, 1.0)
    expectancy_r = _float(perf.get("expectancy_r"), (score_hint - 0.45) * 1.5)
    profit_factor = max(0.0, _float(perf.get("profit_factor"), 1.0 + max(0.0, score_hint - 0.5)))
    slippage_quality = clamp(_float(perf.get("slippage_quality_score"), 0.70), 0.0, 1.0)
    expectancy_component = clamp((expectancy_r + 0.20) / 0.80, 0.0, 1.0)
    profit_factor_component = clamp((profit_factor - 0.80) / 0.70, 0.0, 1.0)
    slippage_adjusted_score = clamp(
        (score_hint * 0.42) + (expectancy_component * 0.28) + (profit_factor_component * 0.18) + (slippage_quality * 0.12),
        0.0,
        1.0,
    )
    promotion_score = clamp(
        slippage_adjusted_score - max(0.0, (0.60 - slippage_quality) * 0.20),
        0.0,
        1.0,
    )
    return ShadowVariantScore(
        variant_id=str(payload.get("variant_id") or ""),
        promotion_score=float(promotion_score),
        expectancy_r=float(expectancy_r),
        profit_factor=float(profit_factor),
        slippage_adjusted_score=float(slippage_adjusted_score),
        promoted=bool(promotion_score >= promotion_threshold),
    )


def build_execution_minute_profile(
    *,
    now_utc: datetime | None,
    runtime: dict[str, Any] | None = None,
    management_feedback: dict[str, Any] | None = None,
) -> ExecutionMinuteProfile:
    runtime_payload = dict(runtime or {})
    feedback_payload = dict(management_feedback or {})
    timestamp = now_utc or datetime.now(tz=UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    minute_of_day = int(timestamp.astimezone(UTC).hour * 60 + timestamp.astimezone(UTC).minute)
    session_name = str(runtime_payload.get("session_name") or runtime_payload.get("current_session_name") or "").upper()
    spread_score = clamp(_float(runtime_payload.get("spread_quality_score"), 0.70), 0.0, 1.0)
    slippage_score = clamp(
        _float(runtime_payload.get("slippage_quality_score"), _float(runtime_payload.get("execution_quality_score"), 0.70)),
        0.0,
        1.0,
    )
    fill_score = clamp(
        _float(feedback_payload.get("active_management_ratio"), _float(feedback_payload.get("fill_quality_score"), 0.65)),
        0.0,
        1.0,
    )
    quality_score = clamp((spread_score * 0.34) + (slippage_score * 0.42) + (fill_score * 0.24), 0.0, 1.0)
    size_multiplier = clamp(0.75 + (quality_score * 0.50), 0.65, 1.25)
    if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} and quality_score >= 0.72:
        size_multiplier = clamp(size_multiplier + 0.05, 0.65, 1.30)
    state = "CLEAN" if quality_score >= 0.76 else "ROUGH" if quality_score <= 0.45 else "MIXED"
    return ExecutionMinuteProfile(
        minute_of_day=minute_of_day,
        session_name=session_name or "UNKNOWN",
        quality_score=float(quality_score),
        size_multiplier=float(size_multiplier),
        state=state,
        spread_score=float(spread_score),
        slippage_score=float(slippage_score),
        fill_score=float(fill_score),
    )
