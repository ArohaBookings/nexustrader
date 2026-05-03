from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
import math

UTC = timezone.utc


DATA_PROVIDERS: tuple[dict[str, Any], ...] = (
    {"id": "mt5", "label": "MT5 Bridge", "aliases": ("mt5", "metatrader", "bridge", "broker")},
    {"id": "polygon", "label": "Polygon.io", "aliases": ("polygon", "polygon.io")},
    {"id": "twelve_data", "label": "Twelve Data", "aliases": ("twelve_data", "twelvedata", "twelve data")},
    {"id": "finnhub", "label": "Finnhub", "aliases": ("finnhub",)},
    {"id": "alpha_vantage", "label": "Alpha Vantage", "aliases": ("alpha_vantage", "alphavantage", "alpha vantage")},
    {"id": "binance", "label": "Binance", "aliases": ("binance",)},
    {"id": "bybit", "label": "Bybit", "aliases": ("bybit",)},
    {"id": "tradingeconomics", "label": "TradingEconomics", "aliases": ("tradingeconomics", "trading_economics")},
    {"id": "newsapi", "label": "NewsAPI", "aliases": ("newsapi", "news_api")},
    {"id": "cryptopanic", "label": "CryptoPanic", "aliases": ("cryptopanic", "crypto_panic")},
    {"id": "hyperliquid", "label": "Hyperliquid", "aliases": ("hyperliquid",)},
)


@dataclass(frozen=True)
class InstitutionalApexInputs:
    health: Mapping[str, Any]
    stats: Mapping[str, Any]
    symbols: Sequence[Mapping[str, Any]]
    opportunities: Sequence[Mapping[str, Any]] = ()
    open_trades: Sequence[Mapping[str, Any]] = ()
    events: Sequence[Mapping[str, Any]] = ()
    learning: Mapping[str, Any] | None = None
    self_heal_status: Mapping[str, Any] | None = None
    risk_config: Mapping[str, Any] | None = None
    no_trade_scoreboard: Mapping[str, Any] | None = None
    active_blockers: Mapping[str, Any] | None = None


def build_institutional_apex_snapshot(
    *,
    health: Mapping[str, Any],
    stats: Mapping[str, Any],
    symbols: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]] = (),
    open_trades: Sequence[Mapping[str, Any]] = (),
    events: Sequence[Mapping[str, Any]] = (),
    learning: Mapping[str, Any] | None = None,
    self_heal_status: Mapping[str, Any] | None = None,
    risk_config: Mapping[str, Any] | None = None,
    no_trade_scoreboard: Mapping[str, Any] | None = None,
    active_blockers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inputs = InstitutionalApexInputs(
        health=health,
        stats=stats,
        symbols=symbols,
        opportunities=opportunities,
        open_trades=open_trades,
        events=events,
        learning=learning,
        self_heal_status=self_heal_status,
        risk_config=risk_config,
        no_trade_scoreboard=no_trade_scoreboard,
        active_blockers=active_blockers,
    )
    funded = _build_funded_mission(inputs)
    mt5 = _build_mt5_bridge(inputs)
    market = _build_market_mastery(inputs)
    fusion = _build_data_fusion(inputs, mt5)
    overfit = _build_anti_overfit_gate(inputs)
    repair = _build_self_repair(inputs, funded)
    scaling = _build_scaling(inputs, funded, overfit, repair)
    execution = _build_execution_quality(inputs)
    grade = _weighted_grade(
        (
            (market["score"], 0.26),
            (fusion["consensus_score"], 0.18),
            (overfit["score"], 0.18),
            (repair["score"], 0.15),
            (scaling["score"], 0.14),
            (execution["score"], 0.09),
        )
    )
    readiness = _readiness(funded, repair, overfit, fusion, grade)
    summary = _summary_for(readiness, funded, repair, overfit)
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "system_grade": grade,
        "grade_pct": round(grade * 100.0, 2),
        "readiness": readiness,
        "summary": summary,
        "mt5_bridge": mt5,
        "funded_mission": funded,
        "market_mastery": market,
        "data_fusion": fusion,
        "anti_overfit": overfit,
        "self_repair": repair,
        "scaling": scaling,
        "execution": execution,
        "telegram_brief": _telegram_brief(readiness, funded, repair, overfit, fusion, scaling),
        "operator_limits": [
            "No direct Telegram trade placement.",
            "No AI-driven risk/aggression increases.",
            "No parameter promotion unless recent and validation gates clear.",
            "Drawdown, stale MT5 data, broker errors, funded floors, and kill switch override frequency.",
        ],
    }


def _build_mt5_bridge(inputs: InstitutionalApexInputs) -> dict[str, Any]:
    broker = _record(inputs.health.get("broker_connectivity"))
    snapshot = _record(inputs.stats.get("latest_account_snapshot"))
    account_scaling = _record(inputs.stats.get("account_scaling"))
    account = _text(broker.get("account") or snapshot.get("account") or account_scaling.get("account"), "")
    terminal_connected = _bool_or_none(broker.get("terminal_connected"))
    terminal_trade_allowed = _bool_or_none(broker.get("terminal_trade_allowed"))
    mql_trade_allowed = _bool_or_none(broker.get("mql_trade_allowed"))
    equity = _first_number(snapshot, account_scaling, "equity", default=0.0)
    balance = _first_number(snapshot, account_scaling, "balance", default=0.0)
    free_margin = _first_number(snapshot, account_scaling, "free_margin", default=0.0)
    latest_symbol = _text(snapshot.get("symbol_key") or snapshot.get("symbol"), "")
    return {
        "connected": bool(terminal_connected) if terminal_connected is not None else bool(account and equity > 0.0),
        "account": account,
        "magic": int(_number(broker.get("magic") or snapshot.get("magic"), 0.0)),
        "terminal_connected": terminal_connected,
        "terminal_trade_allowed": terminal_trade_allowed,
        "mql_trade_allowed": mql_trade_allowed,
        "latest_symbol": latest_symbol,
        "equity": equity,
        "balance": balance,
        "free_margin": free_margin,
        "floating_pnl": _number(snapshot.get("floating_pnl"), 0.0),
        "open_positions": int(_number(snapshot.get("total_open_positions") or inputs.stats.get("open_positions"), 0.0)),
        "source": "mt5_bridge" if account or equity > 0.0 else "bridge_snapshot_missing",
    }


def _build_funded_mission(inputs: InstitutionalApexInputs) -> dict[str, Any]:
    risk_config = _record(inputs.risk_config)
    funded_config = _record(risk_config.get("funded"))
    risk_state = _record(inputs.stats.get("risk_state"))
    snapshot = _record(inputs.stats.get("latest_account_snapshot"))
    scaling = _record(inputs.stats.get("account_scaling"))
    broker = _record(inputs.health.get("broker_connectivity"))
    descriptor = " ".join(
        str(value)
        for value in (
            funded_config.get("group"),
            funded_config.get("provider"),
            funded_config.get("phase"),
            broker.get("account"),
            snapshot.get("account"),
        )
        if value is not None
    ).lower()
    auto_detected = any(token in descriptor for token in ("funded", "prop", "challenge", "evaluation", "verification"))
    enabled = bool(funded_config.get("enabled", auto_detected))
    phase = _text(funded_config.get("phase"), "evaluation")
    group = _text(funded_config.get("group") or funded_config.get("provider"), "custom")
    equity = _first_number(snapshot, scaling, "equity", default=0.0)
    balance = _first_number(snapshot, scaling, "balance", default=equity)
    starting_balance = _number(
        funded_config.get("starting_balance")
        or funded_config.get("initial_balance")
        or risk_state.get("day_start_equity")
        or balance
        or equity,
        equity,
    )
    day_start = _number(risk_state.get("day_start_equity") or inputs.stats.get("day_start_equity"), equity or starting_balance)
    day_high = max(day_start, _number(risk_state.get("day_high_equity") or inputs.stats.get("day_high_equity"), day_start))
    daily_limit_pct = _number(funded_config.get("daily_loss_limit_pct"), 0.05)
    overall_limit_pct = _number(funded_config.get("overall_drawdown_limit_pct"), 0.10)
    target_pct = _number(funded_config.get("profit_target_pct"), 0.08)
    guard_buffer_pct = _number(funded_config.get("guard_buffer_pct"), 0.02)
    daily_dd_pct = max(0.0, _number(inputs.stats.get("daily_dd_pct_live") or risk_state.get("daily_dd_pct_live"), 0.0))
    overall_dd_pct = max(0.0, _number(risk_state.get("absolute_drawdown_pct") or inputs.stats.get("absolute_drawdown_pct"), 0.0))
    daily_limit_usd = max(0.0, day_high * daily_limit_pct)
    overall_limit_usd = max(0.0, starting_balance * overall_limit_pct)
    daily_loss_used_usd = max(0.0, day_high - equity) if equity > 0.0 and day_high > 0.0 else 0.0
    overall_loss_used_usd = max(0.0, starting_balance - equity) if equity > 0.0 and starting_balance > 0.0 else 0.0
    daily_buffer_usd = max(0.0, daily_limit_usd - daily_loss_used_usd)
    overall_buffer_usd = max(0.0, overall_limit_usd - overall_loss_used_usd)
    target_equity = starting_balance * (1.0 + target_pct) if starting_balance > 0.0 else 0.0
    needed_to_pass = max(0.0, target_equity - equity)
    pass_progress_pct = 0.0
    if target_equity > starting_balance and equity > 0.0:
        pass_progress_pct = _clamp((equity - starting_balance) / (target_equity - starting_balance), 0.0, 1.5)
    daily_buffer_pct = daily_buffer_usd / max(day_high, 1e-9)
    overall_buffer_pct = overall_buffer_usd / max(starting_balance, 1e-9)
    buffer_quality = min(
        daily_buffer_usd / max(daily_limit_usd, 1e-9) if daily_limit_usd > 0.0 else 1.0,
        overall_buffer_usd / max(overall_limit_usd, 1e-9) if overall_limit_usd > 0.0 else 1.0,
    )
    risk_throttle = _clamp(0.55 + (_clamp(buffer_quality, 0.0, 1.0) * 0.55), 0.0, 1.0)
    current_daily_state = _text(inputs.health.get("current_daily_state"), "")
    if not enabled:
        status = "disabled"
        guard_reason = "funded_mode_disabled"
    elif current_daily_state == "DAILY_HARD_STOP" or buffer_quality <= max(0.02, guard_buffer_pct * 0.5):
        status = "hard_stop"
        guard_reason = "funded_buffer_exhausted"
        risk_throttle = 0.0
    elif needed_to_pass <= 0.0 and target_equity > 0.0:
        status = "passed"
        guard_reason = "target_reached_protect_pass"
        risk_throttle = min(risk_throttle, 0.55)
    elif pass_progress_pct >= 0.80:
        status = "protect_pass"
        guard_reason = "near_target_protect_buffer"
        risk_throttle = min(risk_throttle, 0.75)
    else:
        status = "active"
        guard_reason = "inside_funded_caps"
    return {
        "enabled": enabled,
        "group": group,
        "phase": phase,
        "status": status,
        "guard_reason": guard_reason,
        "mt5_derived": bool(equity > 0.0 or balance > 0.0),
        "account": {
            "account": _text(broker.get("account") or snapshot.get("account"), ""),
            "equity": equity,
            "balance": balance,
            "free_margin": _first_number(snapshot, scaling, "free_margin", default=0.0),
            "starting_balance": starting_balance,
            "day_start_equity": day_start,
            "day_high_equity": day_high,
        },
        "target_equity": target_equity,
        "needed_to_pass": needed_to_pass,
        "pass_progress_pct": pass_progress_pct,
        "daily_loss_limit_pct": daily_limit_pct,
        "overall_drawdown_limit_pct": overall_limit_pct,
        "profit_target_pct": target_pct,
        "daily_buffer_usd": daily_buffer_usd,
        "daily_buffer_pct": daily_buffer_pct,
        "overall_buffer_usd": overall_buffer_usd,
        "overall_buffer_pct": overall_buffer_pct,
        "daily_dd_pct_live": daily_dd_pct,
        "overall_dd_pct": overall_dd_pct,
        "risk_throttle": risk_throttle,
        "max_risk_per_trade_usd": max(0.0, equity * 0.003 * risk_throttle),
        "max_open_risk_usd": max(0.0, equity * 0.012 * risk_throttle),
    }


def _build_market_mastery(inputs: InstitutionalApexInputs) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw in inputs.symbols:
        symbol = _record(raw)
        quality = _score_value(symbol.get("quality_score") or symbol.get("session_adjusted_score"), 0.45)
        entry = _score_value(symbol.get("entry_timing_score"), quality)
        structure = _score_value(symbol.get("structure_cleanliness_score"), quality)
        confluence = _score_value(symbol.get("confluence"), quality)
        probability = _score_value(symbol.get("probability"), quality)
        regime = _score_value(symbol.get("regime_fit") or symbol.get("regime_confidence"), quality)
        session = _score_value(symbol.get("session_fit"), quality)
        execution = _score_value(symbol.get("execution_quality_fit"), quality)
        behavior = _score_value(symbol.get("pair_behavior_fit"), quality)
        liquidity = _score_value(symbol.get("predicted_liquidity_hunt_score"), structure)
        cross_asset = _cross_asset_score(symbol)
        market_data = _market_data_score(symbol)
        dimensions = {
            "candle": _avg((quality, entry, structure, _score_value(symbol.get("candle_mastery_score"), quality))),
            "smc": _avg((structure, liquidity, _score_value(symbol.get("smc_score"), structure))),
            "order_flow": _avg((liquidity, _score_value(symbol.get("ghost_order_book_proxy"), liquidity), execution)),
            "microstructure": _avg((execution, market_data, behavior)),
            "confluence": _avg((confluence, probability, quality)),
            "regime": _avg((regime, session)),
            "volume_profile": _avg((_score_value(symbol.get("volume_profile_score"), confluence), liquidity)),
            "vwap": _avg((_score_value(symbol.get("vwap_score"), entry), entry)),
            "cross_asset": cross_asset,
        }
        score = _avg(dimensions.values())
        rows.append(
            {
                "symbol": _text(symbol.get("symbol"), "UNKNOWN").upper(),
                "strategy": _text(symbol.get("strategy_pool_winner") or symbol.get("strategy_key"), "strategy_pending"),
                "state": _text(symbol.get("strategy_state") or symbol.get("pair_status"), "unknown"),
                "lane": _text(symbol.get("lane"), ""),
                "score": score,
                "quality_score": quality,
                "probability": probability,
                "regime": _text(symbol.get("regime") or symbol.get("regime_state"), "unclassified"),
                "session_profile": _text(symbol.get("session_priority_profile"), "GLOBAL"),
                "blocker": _text(symbol.get("blocked_reason") or symbol.get("primary_block_reason"), ""),
                "dimensions": dimensions,
                "detected": {
                    "liquidity_sweep_score": liquidity,
                    "fvg_state": _text(symbol.get("last_fvg_zone") or symbol.get("fvg_state"), "unknown"),
                    "bos_choch": _text(symbol.get("bos_choch") or symbol.get("structure_shift"), "unknown"),
                    "order_block": _text(symbol.get("order_block_state") or symbol.get("smc_reason"), "unknown"),
                    "cross_asset_alignment": _text(symbol.get("cross_asset_alignment_state"), "NEUTRAL"),
                },
            }
        )
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return {
        "score": _avg(row["score"] for row in rows) if rows else 0.35,
        "regime": _mode(row["regime"] for row in rows if row["regime"] != "unclassified") or "unclassified",
        "symbols_scanned": len(rows),
        "top_symbols": rows[:8],
        "dimensions": {
            name: _avg(_number((row["dimensions"] or {}).get(name), 0.0) for row in rows) if rows else 0.0
            for name in (
                "candle",
                "smc",
                "order_flow",
                "microstructure",
                "confluence",
                "regime",
                "volume_profile",
                "vwap",
                "cross_asset",
            )
        },
    }


def _build_data_fusion(inputs: InstitutionalApexInputs, mt5: Mapping[str, Any]) -> dict[str, Any]:
    source_status: dict[str, Any] = {}
    symbol_source_text: list[str] = []
    for raw in inputs.symbols:
        symbol = _record(raw)
        for key in (
            "runtime_market_data_source",
            "market_data_source",
            "runtime_market_data_consensus_state",
            "runtime_market_data_mode",
        ):
            value = symbol.get(key)
            if value:
                symbol_source_text.append(str(value).lower())
        diagnostics = _record(symbol.get("runtime_market_data_provider_diagnostics"))
        for key, value in diagnostics.items():
            source_status[str(key).lower()] = value
    providers: list[dict[str, Any]] = []
    for provider in DATA_PROVIDERS:
        provider_id = str(provider["id"])
        aliases = tuple(str(item) for item in provider["aliases"])
        if provider_id == "mt5":
            active = bool(mt5.get("connected"))
            stale = False
            explicit = {"source": mt5.get("source"), "account": mt5.get("account")}
        else:
            explicit = next((source_status.get(alias) for alias in aliases if alias in source_status), None)
            text = " ".join([str(explicit or "").lower(), *symbol_source_text])
            active = any(alias in text for alias in aliases) and any(token in text for token in ("ok", "ready", "active", "native", "consensus", "connected", provider_id))
            stale = any(token in text for token in ("stale", "down", "degraded", "error", "timeout", "missing"))
        status = "active" if active and not stale else "degraded" if stale else "missing"
        explicit_record = _record(explicit)
        providers.append(
            {
                "id": provider_id,
                "label": str(provider["label"]),
                "status": status,
                "latency_ms": _nullable_number(explicit_record.get("latency_ms") or explicit_record.get("latencyMs")),
                "last_seen": _text(explicit_record.get("last_seen") or explicit_record.get("lastSeen"), ""),
            }
        )
    active_count = sum(1 for item in providers if item["status"] == "active")
    degraded_count = sum(1 for item in providers if item["status"] == "degraded")
    consensus_score = _clamp((active_count + degraded_count * 0.45) / max(len(providers), 1), 0.0, 1.0)
    return {
        "consensus_score": consensus_score,
        "active_sources": active_count,
        "degraded_sources": degraded_count,
        "missing_sources": len(providers) - active_count - degraded_count,
        "fallback_ready": bool(active_count >= 2 or mt5.get("connected")),
        "providers": providers,
    }


def _build_anti_overfit_gate(inputs: InstitutionalApexInputs) -> dict[str, Any]:
    learning = _record(inputs.learning)
    optimizer = _record(learning.get("optimizer_summary"))
    rollout = _record(inputs.health.get("current_rollout_stats"))
    recent = _record(rollout.get("last_200") or optimizer.get("last_200"))
    validation = _record(optimizer.get("validation") or optimizer.get("holdout") or optimizer.get("last_100_validation"))
    recent_sample = int(_number(recent.get("trade_count") or optimizer.get("recent_sample_size") or optimizer.get("recent_trades"), 0.0))
    validation_sample = int(_number(validation.get("trade_count") or optimizer.get("validation_sample_size") or optimizer.get("validation_trades"), 0.0))
    recent_expectancy = _number(recent.get("expectancy_r") or recent.get("expectancy") or optimizer.get("recent_expectancy"), 0.0)
    validation_expectancy = _number(validation.get("expectancy_r") or validation.get("expectancy") or optimizer.get("validation_expectancy"), 0.0)
    recent_delta = _number(optimizer.get("recent_expectancy_delta") or optimizer.get("recent_edge_delta"), 0.0)
    validation_delta = _number(optimizer.get("validation_expectancy_delta") or optimizer.get("validation_edge_delta"), 0.0)
    best_rows = [_record(item) for item in _sequence(learning.get("best"))]
    worst_rows = [_record(item) for item in _sequence(learning.get("worst"))]
    if not recent_sample and best_rows:
        recent_sample = max(int(_number(row.get("trade_count"), 0.0)) for row in best_rows)
    if recent_delta == 0.0 and best_rows and worst_rows:
        recent_delta = max(_number(row.get("expectancy_r"), 0.0) for row in best_rows) - max(
            _number(row.get("expectancy_r"), 0.0) for row in worst_rows
        )
    promotion_allowed = bool(
        recent_sample >= 200
        and validation_sample >= 100
        and recent_delta > 0.03
        and validation_delta > 0.03
        and recent_expectancy >= 0.0
        and validation_expectancy >= 0.0
    )
    if promotion_allowed:
        reason = "promotion_gate_cleared"
    elif recent_sample < 200 or validation_sample < 100:
        reason = "insufficient_out_of_sample_evidence"
    else:
        reason = "expectancy_delta_below_3pct_gate"
    return {
        "score": 1.0 if promotion_allowed else 0.72 if recent_sample >= 200 and validation_sample >= 100 else 0.38,
        "promotion_allowed": promotion_allowed,
        "improvement_gate_pct": 0.03,
        "recent_sample": recent_sample,
        "validation_sample": validation_sample,
        "recent_expectancy": recent_expectancy,
        "validation_expectancy": validation_expectancy,
        "recent_delta": recent_delta,
        "validation_delta": validation_delta,
        "reason": reason,
    }


def _build_self_repair(inputs: InstitutionalApexInputs, funded: Mapping[str, Any]) -> dict[str, Any]:
    soft: list[dict[str, Any]] = []
    hard: list[dict[str, Any]] = []
    for raw in inputs.symbols:
        symbol = _record(raw)
        reason = _text(symbol.get("blocked_reason") or symbol.get("primary_block_reason"), "")
        if not reason:
            continue
        item = {"symbol": _text(symbol.get("symbol"), "UNKNOWN"), "reason": reason, "age_minutes": 0}
        if _hard_rail(reason):
            hard.append(item)
        else:
            soft.append(item)
    for key, value in _record(inputs.active_blockers).items():
        for blocker in _sequence(value):
            reason = _text(_record(blocker).get("reason") or blocker, "")
            if reason:
                item = {"symbol": str(key), "reason": reason, "age_minutes": 0}
                (hard if _hard_rail(reason) else soft).append(item)
    health_reason = _text(inputs.health.get("current_daily_state_reason"), "")
    if _text(inputs.health.get("current_daily_state"), "") == "DAILY_HARD_STOP" or _hard_rail(health_reason):
        hard.append({"symbol": "SYSTEM", "reason": health_reason or "daily_hard_stop", "age_minutes": 0})
    if _text(funded.get("status"), "") == "hard_stop":
        hard.append({"symbol": "FUNDED", "reason": _text(funded.get("guard_reason"), "funded_hard_stop"), "age_minutes": 0})
    soft_repairable = [item for item in soft if _soft_repairable(_text(item.get("reason"), ""))]
    self_heal = _record(inputs.self_heal_status)
    configured_actions = list(_sequence(self_heal.get("last_self_heal_actions") or self_heal.get("actions")))
    recommended = "refresh_state" if soft_repairable and not hard else "none"
    status = "hard_rail_holds" if hard else "repair_refresh_required" if soft_repairable else "wait_for_validation" if soft else "clear"
    actions = []
    if recommended == "refresh_state":
        actions.append({"action": "refresh_state", "reason": "soft stale/data/api blocker inside repair SLA", "allowed": True})
    actions.extend({"action": "observe_or_shadow_validate", "reason": f"{item['symbol']}: {item['reason']}", "allowed": True} for item in soft if item not in soft_repairable)
    actions.extend({"action": "do_not_override_hard_rail", "reason": f"{item['symbol']}: {item['reason']}", "allowed": False} for item in hard)
    return {
        "status": status,
        "score": 0.0 if hard else 0.62 if soft else 1.0,
        "sla_minutes": 5,
        "soft_blockers": soft[:30],
        "hard_rails": hard[:20],
        "actions": actions[:40],
        "configured_actions": configured_actions[:20],
        "recommended_bridge_action": recommended,
    }


def _build_scaling(
    inputs: InstitutionalApexInputs,
    funded: Mapping[str, Any],
    anti_overfit: Mapping[str, Any],
    repair: Mapping[str, Any],
) -> dict[str, Any]:
    mt5 = _build_mt5_bridge(inputs)
    equity = _number(mt5.get("equity"), 0.0)
    risk_throttle = _number(funded.get("risk_throttle"), 1.0)
    hard_blocked = bool(_sequence(repair.get("hard_rails"))) or _text(funded.get("status"), "") == "hard_stop"
    promotion_allowed = bool(anti_overfit.get("promotion_allowed"))
    if hard_blocked:
        aggression = "locked"
        score = 0.0
    elif _text(funded.get("status"), "") in {"passed", "protect_pass"}:
        aggression = "protect_pass"
        score = min(0.65, risk_throttle)
    elif promotion_allowed:
        aggression = "expand_inside_caps"
        score = min(1.0, risk_throttle)
    else:
        aggression = "measured_validate"
        score = min(0.72, risk_throttle)
    starting_balance = _number(_record(funded.get("account")).get("starting_balance"), equity)
    funding_change = "capital_increase_detected" if equity > starting_balance * 1.25 and starting_balance > 0.0 else "no_material_top_up_detected"
    max_risk = _number(funded.get("max_risk_per_trade_usd"), equity * 0.003 * risk_throttle)
    max_open_risk = _number(funded.get("max_open_risk_usd"), equity * 0.012 * risk_throttle)
    return {
        "score": score,
        "aggression": aggression,
        "equity": equity,
        "throttle": risk_throttle,
        "funding_change": funding_change,
        "base_capital_protected_usd": min(equity, starting_balance) if starting_balance > 0.0 else equity,
        "max_risk_per_trade_usd": max_risk,
        "max_open_risk_usd": max_open_risk,
        "max_open_trades": int(max(0.0, math.floor(max_open_risk / max(max_risk, 1e-9)))) if max_risk > 0.0 else 0,
        "open_risk_pct": _number(inputs.health.get("open_risk_pct"), 0.0),
        "notes": _scaling_notes(aggression, anti_overfit, funded, repair),
    }


def _build_execution_quality(inputs: InstitutionalApexInputs) -> dict[str, Any]:
    open_trades = [_record(item) for item in inputs.open_trades]
    events = [_record(item) for item in inputs.events]
    rejects = [event for event in events if "reject" in _text(event.get("type") or event.get("reason"), "").lower()]
    fills = [event for event in events if any(token in _text(event.get("type"), "").lower() for token in ("fill", "execution", "open", "close"))]
    pnl_values = [_number(item.get("current_pnl"), 0.0) for item in open_trades if item.get("current_pnl") is not None]
    win_rate = sum(1 for value in pnl_values if value > 0.0) / max(len(pnl_values), 1) if pnl_values else 0.0
    reject_rate = len(rejects) / max(len(events), 1) if events else 0.0
    execution_state = _text(inputs.health.get("execution_quality_state"), "GOOD").upper()
    state_penalty = 0.0 if execution_state in {"GOOD", "OK"} else 0.12 if execution_state in {"CAUTION", "FAIR"} else 0.28
    score = _clamp(0.72 + (win_rate * 0.16) - (reject_rate * 0.25) - state_penalty, 0.0, 1.0)
    return {
        "score": score,
        "open_trades": len(open_trades),
        "event_count": len(events),
        "fills": len(fills),
        "rejections": len(rejects),
        "reject_rate": reject_rate,
        "floating_win_rate": win_rate,
        "execution_state": execution_state,
    }


def _readiness(
    funded: Mapping[str, Any],
    repair: Mapping[str, Any],
    anti_overfit: Mapping[str, Any],
    fusion: Mapping[str, Any],
    grade: float,
) -> str:
    status = _text(funded.get("status"), "disabled")
    if status == "hard_stop" or _sequence(repair.get("hard_rails")):
        return "hard_stop"
    if status in {"passed", "protect_pass"}:
        return "protect_funded_pass"
    if not bool(fusion.get("fallback_ready")) or _text(repair.get("status"), "") == "repair_refresh_required":
        return "repair_first"
    if bool(anti_overfit.get("promotion_allowed")) and grade >= 0.72:
        return "expand_inside_caps"
    return "observe_validate"


def _summary_for(readiness: str, funded: Mapping[str, Any], repair: Mapping[str, Any], anti_overfit: Mapping[str, Any]) -> str:
    if readiness == "hard_stop":
        return "Hard rail active. Preserve account; do not override drawdown, funded, stale data, or kill-switch controls."
    if readiness == "protect_funded_pass":
        return "Funded target state is reached or near; protect the pass before any scale-up."
    if not bool(anti_overfit.get("promotion_allowed")):
        return "Execution can continue inside current caps, but optimization promotion is blocked until out-of-sample evidence clears."
    if readiness == "expand_inside_caps":
        return "Evidence and buffers support measured expansion inside current funded and MT5 risk caps."
    if _sequence(repair.get("soft_blockers")):
        return "Soft blockers are active; refresh state or shadow-validate before adding frequency."
    return "Operate inside current caps while telemetry validates edge."


def _telegram_brief(
    readiness: str,
    funded: Mapping[str, Any],
    repair: Mapping[str, Any],
    anti_overfit: Mapping[str, Any],
    fusion: Mapping[str, Any],
    scaling: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            f"Readiness: {readiness}.",
            (
                "Funded: "
                f"{funded.get('status')}; needed {_money(_number(funded.get('needed_to_pass'), 0.0))}; "
                f"daily buffer {_money(_number(funded.get('daily_buffer_usd'), 0.0))}; "
                f"throttle {_pct(_number(funded.get('risk_throttle'), 0.0))}."
            ),
            (
                "Repair: "
                f"{repair.get('status')}; soft {len(_sequence(repair.get('soft_blockers')))}; "
                f"hard {len(_sequence(repair.get('hard_rails')))}; action {repair.get('recommended_bridge_action')}."
            ),
            (
                "Overfit gate: "
                f"{anti_overfit.get('reason')}; recent {anti_overfit.get('recent_sample')}; "
                f"validation {anti_overfit.get('validation_sample')}."
            ),
            f"Data fusion: {_pct(_number(fusion.get('consensus_score'), 0.0))} consensus across {fusion.get('active_sources')} active sources.",
            f"Scaling: {scaling.get('aggression')}; max risk/trade {_money(_number(scaling.get('max_risk_per_trade_usd'), 0.0))}.",
        ]
    )


def _scaling_notes(
    aggression: str,
    anti_overfit: Mapping[str, Any],
    funded: Mapping[str, Any],
    repair: Mapping[str, Any],
) -> list[str]:
    notes: list[str] = []
    if aggression == "locked":
        notes.append("Hard rails active; scaling is disabled.")
    if aggression == "protect_pass":
        notes.append("Funded target is close/reached; protect pass before scaling.")
    if not bool(anti_overfit.get("promotion_allowed")):
        notes.append("Optimization promotion blocked by anti-overfit gate.")
    if _sequence(repair.get("soft_blockers")):
        notes.append("Soft blockers require refresh/validation before frequency expansion.")
    if _number(funded.get("risk_throttle"), 1.0) < 1.0:
        notes.append("Funded throttle is reducing risk from configured base.")
    if not notes:
        notes.append("Expansion remains bounded by funded max risk and open-risk caps.")
    return notes


def _weighted_grade(items: Sequence[tuple[float, float]]) -> float:
    weight = sum(max(0.0, float(item[1])) for item in items)
    if weight <= 0.0:
        return 0.0
    return _clamp(sum(_clamp(score, 0.0, 1.0) * max(0.0, w) for score, w in items) / weight, 0.0, 1.0)


def _market_data_score(symbol: Mapping[str, Any]) -> float:
    state = _text(symbol.get("runtime_market_data_consensus_state"), "").lower()
    if any(token in state for token in ("ready", "ok", "active", "consensus", "native")):
        return 1.0
    if any(token in state for token in ("stale", "degraded", "fallback")):
        return 0.45
    if _text(symbol.get("runtime_market_data_source"), ""):
        return 0.65
    return 0.35


def _cross_asset_score(symbol: Mapping[str, Any]) -> float:
    for key in ("cross_asset_alignment_score", "lead_lag_alignment_score", "shadow_alignment_score"):
        if symbol.get(key) is not None:
            value = _number(symbol.get(key), 0.0)
            return _clamp((value + 1.0) / 2.0 if value < 0.0 else value, 0.0, 1.0)
    state = _text(symbol.get("cross_asset_alignment_state"), "").upper()
    if state == "ALIGNED":
        return 0.82
    if state == "CONTRARY":
        return 0.22
    return 0.50


def _hard_rail(reason: str) -> bool:
    value = reason.lower()
    return any(token in value for token in ("drawdown", "daily_hard", "hard_stop", "loss_limit", "kill", "breach", "funded_buffer"))


def _soft_repairable(reason: str) -> bool:
    value = reason.lower()
    return any(token in value for token in ("stale", "api", "timeout", "disconnect", "gap", "data", "bridge", "sync", "latency", "snapshot"))


def _score_value(value: Any, fallback: float) -> float:
    parsed = _number(value, float("nan"))
    if not math.isfinite(parsed):
        return _clamp(float(fallback), 0.0, 1.0)
    if parsed > 1.0:
        parsed = parsed / 100.0 if parsed > 5.0 else parsed / 5.0
    return _clamp(parsed, 0.0, 1.0)


def _first_number(primary: Mapping[str, Any], fallback: Mapping[str, Any], key: str, *, default: float) -> float:
    for payload in (primary, fallback):
        value = payload.get(key)
        parsed = _number(value, float("nan"))
        if math.isfinite(parsed) and parsed > 0.0:
            return parsed
    return default


def _nullable_number(value: Any) -> float | None:
    parsed = _number(value, float("nan"))
    return parsed if math.isfinite(parsed) else None


def _number(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _text(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text if text else default


def _record(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _avg(values: Any) -> float:
    clean: list[float] = []
    for value in values:
        parsed = _number(value, float("nan"))
        if math.isfinite(parsed):
            clean.append(parsed)
    return sum(clean) / len(clean) if clean else 0.0


def _mode(values: Any) -> str:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value or "").strip()
        if text:
            counts[text] = counts.get(text, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else ""


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"
