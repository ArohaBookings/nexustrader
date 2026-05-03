from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.aggression_runtime import build_event_directive
from src.symbol_universe import normalize_symbol_key, symbol_asset_class
from src.utils import SessionWindow, clamp, ensure_parent


UTC = timezone.utc


@dataclass
class NewsEvent:
    event_id: str
    timestamp: datetime
    title: str
    currency: str
    impact: str
    source: str
    category: str = "general_macro"
    is_major_risk: bool = False


@dataclass
class NewsDecision:
    safe: bool
    reason: str
    next_safe_time: datetime | None
    source: str
    blocked_event: NewsEvent | None = None
    bias_direction: str = "neutral"
    bias_confidence: float = 0.0
    bias_reason: str = ""
    source_confidence: float = 0.0
    authenticity_risk: float = 0.0
    sentiment_extreme: float = 0.0
    crowding_bias: str = "neutral"
    state: str = "NEWS_SAFE"
    fallback_used: bool = False
    decision_confidence: float = 1.0
    source_used: str = ""


@dataclass
class NewsEngine:
    cache_path: Path
    provider: str
    api_base_url: str
    api_key_env: str
    cache_ttl_seconds: int
    block_high_impact: bool
    block_medium_impact: bool
    block_window_minutes_before: int
    block_window_minutes_after: int
    fail_open: bool = False
    enabled: bool = True
    http_get: Callable[[str], str] | None = None
    api_key: str = ""
    fallback_provider: str = ""
    fallback_api_base_url: str = ""
    fallback_api_key_env: str = ""
    fallback_api_key: str = ""
    fallback_session_windows: list[SessionWindow] = field(default_factory=list)
    supplemental_rss_feeds: list[dict[str, Any]] = field(default_factory=list)
    event_playbook_map: dict[str, str] = field(default_factory=dict)
    logger: Any | None = None
    bias_enabled: bool = True
    bias_lookback_minutes: int = 240
    high_prob_entry_threshold: float = 0.70
    log_refresh_seconds: int = 900
    http_timeout_seconds: float = 6.0
    http_retries: int = 2
    stale_cache_max_age_multiplier: int = 12
    user_agent: str = "Mozilla/5.0 (ApexBot News)"
    rss_headline_limit: int = 6
    _last_log_state: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _last_log_time: dict[str, datetime] = field(default_factory=dict, init=False, repr=False)

    def is_safe_to_trade(self, symbol: str, now_utc: datetime) -> tuple[bool, str, datetime | None]:
        decision = self.evaluate(symbol, now_utc)
        return decision.safe, decision.reason, decision.next_safe_time

    def evaluate(self, symbol: str, now_utc: datetime) -> NewsDecision:
        return self.status_snapshot(symbol, now_utc)["decision"]

    def get_bias(self, symbol: str, now_utc: datetime) -> dict[str, Any]:
        snapshot = self.status_snapshot(symbol, now_utc)
        return {
            "direction": str(snapshot.get("news_bias_direction", "neutral")),
            "confidence": float(snapshot.get("news_confidence", 0.0) or 0.0),
            "high_probability": bool(snapshot.get("high_probability_bias", False)),
        }

    def status_snapshot(self, symbol: str, now_utc: datetime) -> dict[str, Any]:
        if not self.enabled:
            decision = NewsDecision(
                True,
                "news_disabled",
                None,
                "disabled",
                bias_direction="neutral",
                bias_confidence=0.0,
                source_confidence=1.0,
                authenticity_risk=0.0,
                sentiment_extreme=0.0,
                crowding_bias="neutral",
                state="NEWS_SAFE",
                decision_confidence=1.0,
                source_used="disabled",
            )
            self._log(symbol, decision)
            return self._status_payload(symbol, now_utc, decision, [], api_ok=False, source_meta={})

        events, source, api_ok, source_meta = self._current_events(now_utc)
        bias_direction, bias_confidence, bias_reason = self._derive_symbol_bias(symbol, events, now_utc)
        relevant_events = [event for event in events if self._is_relevant(symbol, event)]
        source_confidence = self._source_confidence(relevant_events)
        authenticity_risk = self._authenticity_risk(relevant_events)
        sentiment_extreme, crowding_bias = self._sentiment_extreme(symbol, relevant_events, now_utc)
        decision_confidence = self._decision_confidence(
            source_confidence=source_confidence,
            cache_age_seconds=float(source_meta.get("cache_age_seconds", 0.0) or 0.0),
            fallback_used=bool(source_meta.get("fallback_used", False)),
            api_ok=bool(api_ok),
        )
        if api_ok:
            relevant = relevant_events
            for event in sorted(relevant, key=lambda item: item.timestamp):
                if self._should_block_from_api(event, now_utc):
                    next_safe_time = event.timestamp + timedelta(minutes=self.block_window_minutes_after)
                    decision = NewsDecision(
                        False,
                        f"blocked_{event.impact.lower()}_{event.currency.lower()}_{event.title}",
                        next_safe_time,
                        source,
                        event,
                        bias_direction=bias_direction,
                        bias_confidence=bias_confidence,
                        bias_reason=bias_reason,
                        source_confidence=source_confidence,
                        authenticity_risk=authenticity_risk,
                        sentiment_extreme=sentiment_extreme,
                        crowding_bias=crowding_bias,
                        state="NEWS_BLOCKED",
                        fallback_used=bool(source_meta.get("fallback_used", False)),
                        decision_confidence=decision_confidence,
                        source_used=str(source_meta.get("source_used", source or self.provider)),
                    )
                    self._log(symbol, decision)
                    return self._status_payload(symbol, now_utc, decision, relevant, api_ok=api_ok, source_meta=source_meta)
            decision = NewsDecision(
                True,
                "clear",
                None,
                source,
                bias_direction=bias_direction,
                bias_confidence=bias_confidence,
                bias_reason=bias_reason,
                source_confidence=source_confidence,
                authenticity_risk=authenticity_risk,
                sentiment_extreme=sentiment_extreme,
                crowding_bias=crowding_bias,
                state="NEWS_SAFE",
                fallback_used=bool(source_meta.get("fallback_used", False)),
                decision_confidence=decision_confidence,
                source_used=str(source_meta.get("source_used", source or self.provider)),
            )
            self._log(symbol, decision)
            return self._status_payload(symbol, now_utc, decision, relevant, api_ok=api_ok, source_meta=source_meta)

        decision = self._unknown_api_decision(symbol, now_utc, relevant_events=relevant_events, source_meta=source_meta)
        decision.bias_direction = bias_direction
        decision.bias_confidence = bias_confidence
        decision.bias_reason = bias_reason
        decision.source_confidence = source_confidence
        decision.authenticity_risk = authenticity_risk
        decision.sentiment_extreme = sentiment_extreme
        decision.crowding_bias = crowding_bias
        decision.decision_confidence = decision_confidence
        if not decision.source_used:
            decision.source_used = str(source_meta.get("source_used", source or "provider_unavailable"))
        self._log(symbol, decision)
        return self._status_payload(symbol, now_utc, decision, relevant_events, api_ok=api_ok, source_meta=source_meta)

    def _status_payload(
        self,
        symbol: str,
        now_utc: datetime,
        decision: NewsDecision,
        relevant_events: list[NewsEvent],
        *,
        api_ok: bool,
        source_meta: dict[str, Any],
    ) -> dict[str, Any]:
        current = now_utc.astimezone(UTC)
        future_events = sorted(
            [event for event in relevant_events if event.timestamp >= current],
            key=lambda event: event.timestamp,
        )
        recent_events = sorted(
            [
                event
                for event in relevant_events
                if timedelta(minutes=0) <= (current - event.timestamp) <= timedelta(minutes=max(self.block_window_minutes_after, 60))
            ],
            key=lambda event: event.timestamp,
            reverse=True,
        )
        next_event = future_events[0] if future_events else None
        next_event_payload = None
        if next_event is not None:
            next_event_payload = {
                "event_id": next_event.event_id,
                "timestamp": next_event.timestamp.isoformat(),
                "title": next_event.title,
                "currency": next_event.currency,
                "impact": next_event.impact,
                "source": next_event.source,
            }
        cache_reused = bool(source_meta.get("cache_reused", False))
        fallback_used = bool(source_meta.get("fallback_used", False))
        headline_events = self._headline_events(
            future_events=future_events,
            recent_events=recent_events,
            now_utc=current,
        )
        headline_payload = [
            {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "title": event.title,
                "source": event.source,
                "impact": event.impact,
                "currency": event.currency,
                "category": event.category,
            }
            for event in headline_events
        ]
        source_breakdown: dict[str, int] = {}
        category_summary: dict[str, int] = {}
        for event in relevant_events:
            source_name = str(event.source or "unknown").strip() or "unknown"
            source_breakdown[source_name] = int(source_breakdown.get(source_name, 0)) + 1
            category_name = str(event.category or "general_macro").strip() or "general_macro"
            category_summary[category_name] = int(category_summary.get(category_name, 0)) + 1
        rss_headlines = [
            item
            for item in headline_payload
            if "INVESTING" in str(item.get("source") or "").upper()
        ]
        primary_category = "general_macro"
        if category_summary:
            primary_category = max(category_summary.items(), key=lambda item: (item[1], item[0]))[0]
        summary_text = self._headline_summary_text(decision=decision, headlines=headline_payload)
        payload = {
            "decision": decision,
            "news_refresh_at": current.isoformat(),
            "next_macro_event": next_event_payload,
            "event_risk_window_active": bool(not decision.safe and decision.blocked_event is not None),
            "post_news_trade_window_active": bool(recent_events),
            "news_state": str(decision.state),
            "news_source_used": str(decision.source_used or decision.source),
            "news_fallback_used": fallback_used,
            "news_decision_confidence": float(decision.decision_confidence),
            "news_decision_reason": str(decision.reason),
            "news_next_event_time": str(next_event.timestamp.isoformat()) if next_event is not None else "",
            "news_next_event_impact": str(next_event.impact) if next_event is not None else "",
            "news_bias_direction": str(decision.bias_direction),
            "news_confidence": float(decision.bias_confidence),
            "news_headlines": headline_payload,
            "news_source_breakdown": source_breakdown,
            "news_category_summary": category_summary,
            "news_primary_category": str(primary_category),
            "news_secondary_source_used": bool(rss_headlines),
            "news_rss_headlines": rss_headlines,
            "news_rss_headline_count": int(len(rss_headlines)),
            "news_data_quality": (
                "provider"
                if api_ok
                else ("cached_fallback" if cache_reused else ("fallback" if self.fail_open else "degraded"))
            ),
            "public_proxy_availability": {
                "calendar_api": bool(api_ok),
                "headline_bias": bool(self.bias_enabled),
                "cached_calendar": cache_reused,
                "fallback_session_windows": bool(self.fallback_session_windows),
            },
            "macro_event_bias": {
                "direction": str(decision.bias_direction),
                "confidence": float(decision.bias_confidence),
                "reason": str(decision.bias_reason),
                "primary_category": str(primary_category),
            },
            "high_probability_bias": bool(
                decision.safe
                and str(decision.bias_direction).lower() in {"bullish", "bearish"}
                and float(decision.bias_confidence) >= float(self.high_prob_entry_threshold)
            ),
            "session_bias_summary": summary_text,
            "pre_open_news_summary": summary_text,
            "pre_open_risk_notes": {
                "event_blocked": bool(decision.state == "NEWS_BLOCKED"),
                "news_state": str(decision.state),
                "authenticity_risk": float(decision.authenticity_risk),
                "source_confidence": float(decision.source_confidence),
                "crowding_bias": str(decision.crowding_bias),
                "decision_confidence": float(decision.decision_confidence),
                "fallback_used": fallback_used,
            },
            "symbol": str(symbol).upper(),
        }
        directive = build_event_directive(
            symbol=str(symbol).upper(),
            news_snapshot=payload,
            playbook_map=self.event_playbook_map,
        )
        payload["event_directive"] = directive.as_dict()
        payload["event_playbook"] = str(directive.playbook)
        payload["event_base_class"] = str(directive.base_class)
        payload["event_sub_class"] = str(directive.sub_class)
        payload["event_pre_position_allowed"] = bool(directive.pre_position_allowed)
        payload["event_pre_position_window_minutes"] = int(directive.pre_position_window_minutes)
        return payload

    def _headline_events(
        self,
        *,
        future_events: list[NewsEvent],
        recent_events: list[NewsEvent],
        now_utc: datetime,
    ) -> list[NewsEvent]:
        limit = max(1, int(self.rss_headline_limit))
        ordered: list[NewsEvent] = []
        seen: set[str] = set()
        future_cutoff = now_utc + timedelta(hours=4)
        for event in future_events:
            if event.timestamp > future_cutoff:
                continue
            key = str(event.event_id)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(event)
            if len(ordered) >= limit:
                return ordered
        for event in recent_events:
            key = str(event.event_id)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(event)
            if len(ordered) >= limit:
                break
        return ordered

    @staticmethod
    def _headline_summary_text(*, decision: NewsDecision, headlines: list[dict[str, Any]]) -> str:
        base_reason = str(decision.bias_reason or decision.reason or "").strip()
        if not headlines:
            return base_reason or str(decision.reason or "clear")
        snippets: list[str] = []
        for item in headlines[:2]:
            title = str(item.get("title") or "").strip()
            source = str(item.get("source") or "").strip()
            if not title:
                continue
            snippets.append(f"{source}: {title}" if source else title)
        if not snippets:
            return base_reason or str(decision.reason or "clear")
        if base_reason:
            return f"{base_reason} | {'; '.join(snippets)}"
        return "; ".join(snippets)

    def _current_events(self, now_utc: datetime) -> tuple[list[NewsEvent], str, bool, dict[str, Any]]:
        cache = self._load_cache()
        if cache:
            expires_at = cache.get("expires_at")
            if expires_at:
                expiry = datetime.fromisoformat(str(expires_at))
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=UTC)
                if expiry >= now_utc.astimezone(UTC):
                    api_ok = bool(cache.get("api_ok", False))
                    events, source = self._events_from_payload(cache.get("events", []), str(cache.get("source", "cache")))
                    fetched_at = self._parse_timestamp(cache.get("fetched_at"))
                    cache_age_seconds = 0.0
                    if fetched_at is not None:
                        cache_age_seconds = max(0.0, (now_utc.astimezone(UTC) - fetched_at).total_seconds())
                    return events, source, api_ok, {
                        "cache_reused": True,
                        "fallback_used": not api_ok,
                        "cache_age_seconds": cache_age_seconds,
                        "source_used": str(cache.get("source", "cache")),
                    }

        events, source, api_ok = self._refresh_events(now_utc)
        if api_ok:
            self._save_cache(events, source, now_utc, api_ok)
            return events, source, api_ok, {
                "cache_reused": False,
                "fallback_used": False,
                "cache_age_seconds": 0.0,
                "source_used": source,
            }

        if cache:
            fallback = self._stale_cache_fallback(cache=cache, now_utc=now_utc, source=source)
            if fallback is not None:
                events, fallback_source, fallback_meta = fallback
                return events, fallback_source, False, fallback_meta

        self._save_cache(events, source, now_utc, api_ok)
        return events, source, api_ok, {
            "cache_reused": False,
            "fallback_used": True,
            "cache_age_seconds": 0.0,
            "source_used": source,
        }

    def _refresh_events(self, now_utc: datetime) -> tuple[list[NewsEvent], str, bool]:
        provider_events: list[NewsEvent] = []
        provider_source = "provider_unavailable"
        provider_api_ok = False
        providers = [
            (
                str(self.provider or "").strip(),
                str(self.api_base_url or "").strip(),
                str(self.api_key_env or "").strip(),
                str(self.api_key or "").strip(),
            )
        ]
        fallback_provider = str(self.fallback_provider or "").strip()
        if fallback_provider and fallback_provider.lower() != str(self.provider or "").strip().lower():
            providers.append(
                (
                    fallback_provider,
                    str(self.fallback_api_base_url or "").strip(),
                    str(self.fallback_api_key_env or "").strip(),
                    str(self.fallback_api_key or "").strip(),
                )
            )
        for provider_name, api_base_url, api_key_env, configured_key in providers:
            api_key = self._resolve_api_key(api_key_env=api_key_env, configured_key=configured_key)
            if not api_base_url or not api_key:
                continue
            try:
                provider_events = self._fetch_provider_events(
                    now_utc,
                    api_key,
                    provider_name=provider_name,
                    api_base_url=api_base_url,
                )
                provider_source = provider_name
                provider_api_ok = True
                break
            except Exception:
                continue
        supplemental_events = self._fetch_supplemental_events(now_utc)
        merged_events = self._merge_events(provider_events, supplemental_events)
        if provider_api_ok:
            source_name = provider_source
            if supplemental_events:
                source_name = f"{provider_source}+supplemental"
            return merged_events, source_name, True
        if supplemental_events:
            return merged_events, "supplemental_only", False
        if self.fail_open:
            return [], "provider_unavailable_fail_open", False
        return [], "provider_unavailable", False

    def _fetch_supplemental_events(self, now_utc: datetime) -> list[NewsEvent]:
        feeds = list(self.supplemental_rss_feeds or [])
        if not feeds:
            return []
        events: list[NewsEvent] = []
        max_age = now_utc.astimezone(UTC) - timedelta(hours=24)
        future_cutoff = now_utc.astimezone(UTC) + timedelta(hours=6)
        for feed in feeds:
            if isinstance(feed, str):
                feed = {"url": str(feed)}
            if not isinstance(feed, dict) or not bool(feed.get("enabled", True)):
                continue
            url = str(feed.get("url") or "").strip()
            if not url:
                continue
            label = str(feed.get("label") or feed.get("source") or "Investing RSS").strip() or "Investing RSS"
            per_feed_limit = max(1, int(feed.get("max_items", self.rss_headline_limit)))
            try:
                body = self._fetch_url(url)
            except Exception:
                continue
            parsed_events = self._parse_rss_feed_events(body, feed_label=label)
            filtered_events = [
                event
                for event in parsed_events
                if max_age <= event.timestamp <= future_cutoff
            ]
            events.extend(filtered_events[:per_feed_limit])
        return self._merge_events(events)

    def _parse_rss_feed_events(self, body: str, *, feed_label: str) -> list[NewsEvent]:
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            return []
        channel = root.find("channel")
        if channel is None:
            return []
        channel_title = str(channel.findtext("title") or feed_label or "Investing RSS").strip() or str(feed_label or "Investing RSS")
        events: list[NewsEvent] = []
        for item in channel.findall("item"):
            title = str(item.findtext("title") or "").strip()
            if not title:
                continue
            description = str(item.findtext("description") or "").strip()
            parsed_time = self._parse_timestamp(
                item.findtext("pubDate") or item.findtext("published") or item.findtext("updated")
            )
            if parsed_time is None:
                continue
            event_id = str(
                item.findtext("guid")
                or item.findtext("link")
                or f"{channel_title}-{parsed_time.isoformat()}-{title}"
            ).strip()
            impact = self._rss_impact(title=title, channel_title=channel_title, description=description)
            events.append(
                NewsEvent(
                    event_id=event_id,
                    timestamp=parsed_time,
                    title=title,
                    currency=self._infer_rss_currency(title=title, description=description),
                    impact=impact,
                    source=str(channel_title),
                    category=self._categorize_event(
                        title=title,
                        source=str(channel_title),
                        currency=self._infer_rss_currency(title=title, description=description),
                        description=description,
                    ),
                    is_major_risk=bool(
                        impact == "HIGH"
                        or self._looks_high_impact_title(title)
                        or self._geopolitical_risk_score(f"{title} {description}") > 0.0
                    ),
                )
            )
        return events

    @staticmethod
    def _merge_events(*event_groups: list[NewsEvent]) -> list[NewsEvent]:
        merged: list[NewsEvent] = []
        seen: set[str] = set()
        for group in event_groups:
            for event in group:
                key = str(event.event_id or f"{event.source}:{event.timestamp.isoformat()}:{event.title}").strip()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(event)
        merged.sort(key=lambda item: item.timestamp, reverse=True)
        return merged

    @staticmethod
    def _rss_impact(*, title: str, channel_title: str, description: str) -> str:
        combined = f"{channel_title} {title} {description}".upper()
        if "ECONOMIC INDICATORS" in combined or "BREAKING NEWS" in combined:
            return "HIGH"
        if any(token in combined for token in ("FOREX", "COMMODITIES", "FUTURES", "CENTRAL BANK", "FED", "ECB", "BOJ", "BOE")):
            return "MED"
        return "HIGH" if NewsEngine._looks_high_impact_title(title) else "LOW"

    @staticmethod
    def _infer_rss_currency(*, title: str, description: str) -> str:
        text = f"{title} {description}".upper()
        if any(token in text for token in ("RBA", "AUD", "AUSSIE", "AUSTRALIAN DOLLAR")):
            return "AUD"
        if any(token in text for token in ("RBNZ", "NZD", "KIWI")):
            return "NZD"
        if any(token in text for token in ("BOC", "CAD", "LOONIE")):
            return "CAD"
        if any(token in text for token in ("SNB", "CHF", "FRANC")):
            return "CHF"
        if any(token in text for token in ("PBOC", "CNY", "YUAN", "CHINA")):
            return "CNY"
        return NewsEngine._infer_newsapi_currency(text)

    def _fetch_provider_events(
        self,
        now_utc: datetime,
        api_key: str,
        *,
        provider_name: str | None = None,
        api_base_url: str | None = None,
    ) -> list[NewsEvent]:
        resolved_provider = str(provider_name or self.provider).strip().lower()
        resolved_base_url = str(api_base_url or self.api_base_url).strip()
        if resolved_provider == "newsapi" or "newsapi.org" in resolved_base_url.lower():
            return self._fetch_newsapi_events(now_utc, api_key)
        if resolved_provider == "finnhub" or "finnhub.io" in resolved_base_url.lower():
            return self._fetch_finnhub_events(now_utc, api_key, api_base_url=resolved_base_url)
        if resolved_provider in {"tradingeconomics", "trading_economics"} or "tradingeconomics.com" in resolved_base_url.lower():
            return self._fetch_tradingeconomics_events(now_utc, api_key, api_base_url=resolved_base_url)

        start = now_utc.date().isoformat()
        end = (now_utc.date() + timedelta(days=2)).isoformat()
        params = {
            "from": start,
            "to": end,
            "apikey": api_key,
        }
        query = urlencode(params)
        url = f"{resolved_base_url}?{query}"
        body = self._fetch_url(url)
        payload = json.loads(body)
        if isinstance(payload, dict) and (payload.get("status") == "error" or payload.get("error")):
            raise RuntimeError(str(payload.get("message") or payload.get("error") or "news_provider_error"))
        items = payload if isinstance(payload, list) else payload.get("data", [])
        events: list[NewsEvent] = []
        for item in items:
            event = self._parse_provider_event(item, resolved_provider)
            if event:
                events.append(event)
        return events

    def _fetch_newsapi_events(self, now_utc: datetime, api_key: str) -> list[NewsEvent]:
        params = {
            "q": "\"FOMC\" OR \"Fed\" OR \"CPI\" OR \"PCE\" OR \"NFP\" OR \"payrolls\" OR \"GDP\" OR \"ECB\" OR \"BOE\" OR \"BOJ\"",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
            "from": (now_utc - timedelta(hours=12)).isoformat(),
            "apiKey": api_key,
        }
        url = f"{self.api_base_url}?{urlencode(params)}"
        body = self._fetch_url(url)
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise RuntimeError("newsapi_invalid_payload")
        if payload.get("status") == "error":
            raise RuntimeError(str(payload.get("message") or payload.get("code") or "newsapi_error"))
        items = payload.get("articles", []) if isinstance(payload, dict) else []
        events: list[NewsEvent] = []
        for item in items:
            event = self._parse_newsapi_article(item)
            if event:
                events.append(event)
        return events

    def _fetch_finnhub_events(self, now_utc: datetime, api_key: str, *, api_base_url: str) -> list[NewsEvent]:
        params = {
            "from": now_utc.date().isoformat(),
            "to": (now_utc.date() + timedelta(days=2)).isoformat(),
            "token": api_key,
        }
        url = f"{api_base_url}?{urlencode(params)}"
        body = self._fetch_url(url)
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise RuntimeError("finnhub_invalid_payload")
        if payload.get("error"):
            raise RuntimeError(str(payload.get("error") or "finnhub_error"))
        items = (
            payload.get("economicCalendar")
            or payload.get("calendar")
            or payload.get("data")
            or []
        )
        events: list[NewsEvent] = []
        for item in items:
            event = self._parse_finnhub_event(item)
            if event:
                events.append(event)
        return events

    def _fetch_tradingeconomics_events(self, now_utc: datetime, api_key: str, *, api_base_url: str) -> list[NewsEvent]:
        base_url = str(api_base_url or self.api_base_url).strip() or "https://api.tradingeconomics.com/calendar"
        params = {
            "c": api_key,
            "f": "json",
            "importance": 2,
        }
        separator = "&" if "?" in base_url else "?"
        url = f"{base_url}{separator}{urlencode(params)}"
        body = self._fetch_url(url)
        payload = json.loads(body)
        if not isinstance(payload, list):
            raise RuntimeError("tradingeconomics_invalid_payload")
        horizon_start = now_utc.astimezone(UTC) - timedelta(hours=12)
        horizon_end = now_utc.astimezone(UTC) + timedelta(days=2)
        events: list[NewsEvent] = []
        for item in payload:
            event = self._parse_tradingeconomics_event(item)
            if event and horizon_start <= event.timestamp <= horizon_end:
                events.append(event)
        return events

    def _parse_provider_event(self, item: dict[str, Any], source: str) -> NewsEvent | None:
        if not isinstance(item, dict):
            return None
        raw_time = item.get("date") or item.get("datetime") or item.get("time")
        parsed_time = self._parse_timestamp(raw_time)
        if parsed_time is None:
            return None
        currency = str(item.get("currency") or item.get("country") or item.get("symbol") or "ALL").upper()
        title = str(item.get("event") or item.get("title") or item.get("name") or "event").strip() or "event"
        impact = self._normalize_impact(item.get("impact") or item.get("importance") or item.get("impactLevel"))
        major = any(token in title.upper() for token in ("FOMC", "RATE", "NFP", "CPI", "PCE", "GDP"))
        event_id = str(item.get("id") or f"{currency}-{parsed_time.isoformat()}-{title}")
        return NewsEvent(
            event_id=event_id,
            timestamp=parsed_time,
            title=title,
            currency=currency,
            impact=impact,
            source=source,
            category=self._categorize_event(title=title, source=source, currency=currency),
            is_major_risk=major,
        )

    def _parse_tradingeconomics_event(self, item: dict[str, Any]) -> NewsEvent | None:
        if not isinstance(item, dict):
            return None
        raw_time = item.get("Date") or item.get("LastUpdate") or item.get("date")
        parsed_time = self._parse_timestamp(raw_time)
        if parsed_time is None:
            return None
        title = str(item.get("Event") or item.get("Category") or item.get("Title") or "event").strip() or "event"
        country = str(item.get("Country") or item.get("currency") or item.get("Currency") or "ALL").strip()
        currency = self._currency_from_country(country)
        impact = self._normalize_impact(item.get("Importance") or item.get("importance") or item.get("Impact"))
        event_id = str(
            item.get("CalendarId")
            or item.get("id")
            or f"te-{currency}-{parsed_time.isoformat()}-{title}"
        ).strip()
        description = str(item.get("Source") or item.get("URL") or item.get("Reference") or "").strip()
        return NewsEvent(
            event_id=event_id,
            timestamp=parsed_time,
            title=title,
            currency=currency,
            impact=impact,
            source="tradingeconomics",
            category=self._categorize_event(title=title, source="tradingeconomics", currency=currency, description=description),
            is_major_risk=bool(
                impact == "HIGH"
                or any(token in title.upper() for token in ("FOMC", "RATE", "CPI", "PCE", "NFP", "GDP", "PMI"))
            ),
        )

    def _should_block_from_api(self, event: NewsEvent, now_utc: datetime) -> bool:
        if not self.block_high_impact:
            return False
        if event.impact != "HIGH":
            return False
        start = event.timestamp - timedelta(minutes=self.block_window_minutes_before)
        end = event.timestamp + timedelta(minutes=self.block_window_minutes_after)
        return start <= now_utc.astimezone(UTC) <= end

    def _unknown_api_decision(
        self,
        symbol: str,
        now_utc: datetime,
        *,
        relevant_events: list[NewsEvent],
        source_meta: dict[str, Any],
    ) -> NewsDecision:
        blocking_event = None
        for event in sorted(relevant_events, key=lambda item: item.timestamp):
            if self._should_block_from_api(event, now_utc):
                blocking_event = event
                break
        if blocking_event is not None:
            next_safe_time = blocking_event.timestamp + timedelta(minutes=self.block_window_minutes_after)
            return NewsDecision(
                False,
                f"blocked_{blocking_event.impact.lower()}_{blocking_event.currency.lower()}_{blocking_event.title}",
                next_safe_time,
                str(source_meta.get("source_used", "provider_unavailable")),
                blocked_event=blocking_event,
                state="NEWS_BLOCKED",
                fallback_used=True,
                source_used=str(source_meta.get("source_used", "provider_unavailable")),
            )
        if self._allow_unknown_news(symbol):
            return NewsDecision(
                True,
                "news_caution_crypto_allowed",
                None,
                "provider_unavailable",
                state="NEWS_CAUTION",
                fallback_used=True,
                source_used=str(source_meta.get("source_used", "provider_unavailable")),
            )
        if self.fail_open:
            return NewsDecision(
                True,
                "news_caution_fail_open",
                None,
                "provider_unavailable",
                state="NEWS_CAUTION",
                fallback_used=True,
                source_used=str(source_meta.get("source_used", "provider_unavailable")),
            )
        recent_relevant = any(
            abs((now_utc.astimezone(UTC) - event.timestamp).total_seconds()) <= max(self.block_window_minutes_after, 60) * 60
            for event in relevant_events
        )
        for window in self.fallback_session_windows:
            if window.enabled and window.contains(now_utc):
                reason = "news_caution_cached_recent" if recent_relevant else "news_caution_provider_unavailable"
                return NewsDecision(
                    True,
                    reason,
                    None,
                    "provider_unavailable",
                    state="NEWS_CAUTION",
                    fallback_used=True,
                    source_used=str(source_meta.get("source_used", "provider_unavailable")),
                )
        if recent_relevant:
            return NewsDecision(
                True,
                "news_caution_recent_event_context",
                None,
                "provider_unavailable",
                state="NEWS_CAUTION",
                fallback_used=True,
                source_used=str(source_meta.get("source_used", "provider_unavailable")),
            )
        return NewsDecision(
            True,
            "news_degraded_outside_main_session",
            None,
            "provider_unavailable",
            state="NEWS_SAFE",
            fallback_used=True,
            source_used=str(source_meta.get("source_used", "provider_unavailable")),
        )

    def _stale_cache_fallback(
        self,
        *,
        cache: dict[str, Any],
        now_utc: datetime,
        source: str,
    ) -> tuple[list[NewsEvent], str, dict[str, Any]] | None:
        fetched_at = self._parse_timestamp(cache.get("fetched_at"))
        if fetched_at is None:
            return None
        cache_age_seconds = max(0.0, (now_utc.astimezone(UTC) - fetched_at).total_seconds())
        max_age_seconds = max(
            self.cache_ttl_seconds * max(1, int(self.stale_cache_max_age_multiplier)),
            (self.block_window_minutes_before + self.block_window_minutes_after + 60) * 60,
        )
        if cache_age_seconds > max_age_seconds:
            return None
        events, cached_source = self._events_from_payload(cache.get("events", []), str(cache.get("source", "cache")))
        return events, f"{source}+stale_cache", {
            "cache_reused": True,
            "fallback_used": True,
            "cache_age_seconds": cache_age_seconds,
            "source_used": f"{cached_source}:stale_cache",
        }

    @staticmethod
    def _decision_confidence(
        *,
        source_confidence: float,
        cache_age_seconds: float,
        fallback_used: bool,
        api_ok: bool,
    ) -> float:
        confidence = clamp(float(source_confidence), 0.0, 1.0)
        if not api_ok:
            confidence = min(confidence, 0.68 if fallback_used else 0.58)
        if cache_age_seconds > 0:
            age_penalty = min(0.30, cache_age_seconds / 21600.0)
            confidence = max(0.20, confidence - age_penalty)
        return clamp(confidence, 0.0, 1.0)

    def _is_relevant(self, symbol: str, event: NewsEvent) -> bool:
        relevant_currencies = self._relevant_currencies(symbol)
        return event.currency in relevant_currencies or event.currency == "ALL" or event.is_major_risk

    def _relevant_currencies(self, symbol: str) -> set[str]:
        normalized = normalize_symbol_key(symbol)
        asset_class = symbol_asset_class(normalized)
        if len(normalized) == 6 and asset_class == "forex":
            return {normalized[:3], normalized[3:6]}
        if normalized in {"XAUUSD", "XAGUSD"}:
            return {"USD"}
        if asset_class in {"crypto", "equity", "index"}:
            return {"USD"}
        if normalized == "USOIL":
            return {"USD", "CAD"}
        return {"USD"}

    @staticmethod
    def _allow_unknown_news(symbol: str) -> bool:
        return symbol_asset_class(symbol) == "crypto"

    def _parse_newsapi_article(self, item: dict[str, Any]) -> NewsEvent | None:
        if not isinstance(item, dict):
            return None
        parsed_time = self._parse_timestamp(item.get("publishedAt"))
        if parsed_time is None:
            return None
        title = str(item.get("title") or item.get("description") or "macro headline").strip()
        if not title:
            title = "macro headline"
        currency = self._infer_newsapi_currency(title)
        impact = "HIGH" if self._looks_high_impact_title(title) else "LOW"
        source_name = "newsapi"
        source_payload = item.get("source")
        if isinstance(source_payload, dict) and source_payload.get("name"):
            source_name = str(source_payload["name"])
        return NewsEvent(
            event_id=str(item.get("url") or f"{currency}-{parsed_time.isoformat()}-{title}"),
            timestamp=parsed_time,
            title=title,
            currency=currency,
            impact=impact,
            source=source_name,
            category=self._categorize_event(title=title, source=source_name, currency=currency),
            is_major_risk=self._looks_high_impact_title(title),
        )

    def _parse_finnhub_event(self, item: dict[str, Any]) -> NewsEvent | None:
        if not isinstance(item, dict):
            return None
        parsed_time = self._parse_timestamp(item.get("date") or item.get("time") or item.get("datetime"))
        if parsed_time is None:
            return None
        title = str(item.get("event") or item.get("headline") or item.get("indicator") or "economic event").strip()
        if not title:
            title = "economic event"
        currency = str(item.get("currency") or "").strip().upper()
        if not currency:
            currency = self._currency_from_country(str(item.get("country") or item.get("region") or "ALL"))
        impact = self._normalize_impact(item.get("impact") or item.get("importance") or item.get("impactLevel"))
        return NewsEvent(
            event_id=str(item.get("id") or f"finnhub-{currency}-{parsed_time.isoformat()}-{title}"),
            timestamp=parsed_time,
            title=title,
            currency=currency or "ALL",
            impact=impact,
            source="finnhub",
            category=self._categorize_event(title=title, source="finnhub", currency=currency or "ALL"),
            is_major_risk=self._looks_high_impact_title(title) or impact == "HIGH",
        )

    def _derive_symbol_bias(self, symbol: str, events: list[NewsEvent], now_utc: datetime) -> tuple[str, float, str]:
        if not self.bias_enabled or not events:
            return "neutral", 0.0, "bias_disabled"
        horizon = now_utc.astimezone(UTC) - timedelta(minutes=max(5, self.bias_lookback_minutes))
        scoped = [event for event in events if event.timestamp >= horizon and self._is_relevant(symbol, event)]
        if not scoped:
            return "neutral", 0.0, "no_recent_relevant_events"

        currency_scores: dict[str, float] = {}
        risk_score = 0.0
        symbol_key = "".join(char for char in str(symbol or "").upper() if char.isalnum())
        for event in scoped:
            event_score = self._event_direction_score(event.title)
            impact_weight = 1.0 if event.impact == "HIGH" else 0.5 if event.impact == "MED" else 0.25
            oil_event_score = self._oil_supply_direction_score(event.title)
            if symbol_key.startswith("USOIL") and abs(oil_event_score) > 0.0:
                currency_scores["OIL"] = currency_scores.get("OIL", 0.0) + (oil_event_score * impact_weight)
            geopolitical_risk = self._geopolitical_risk_score(event.title)
            if geopolitical_risk > 0.0 and symbol_key.startswith(("XAUUSD", "GOLD", "BTCUSD", "XBTUSD", "USOIL", "XTIUSD", "WTI", "CL")):
                risk_score += geopolitical_risk * impact_weight
            if event.currency == "ALL":
                risk_score += event_score * impact_weight
                continue
            currency_scores[event.currency] = currency_scores.get(event.currency, 0.0) + (event_score * impact_weight)
            if event.is_major_risk:
                risk_score += event_score * 0.25

        symbol_score = self._symbol_bias_score(symbol, currency_scores, risk_score)
        confidence = min(1.0, abs(symbol_score))
        if symbol_key.startswith("USOIL"):
            confidence = clamp(
                max(confidence, min(0.90, abs(float(currency_scores.get("OIL", 0.0) or 0.0)) * 0.45)),
                0.0,
                1.0,
            )
        if symbol_score > 0.2:
            return "bullish", confidence, "macro_bias_positive"
        if symbol_score < -0.2:
            return "bearish", confidence, "macro_bias_negative"
        return "neutral", confidence, "macro_bias_mixed"

    def _source_confidence(self, events: list[NewsEvent]) -> float:
        if not events:
            return 0.5
        scores = [self._source_score(event.source) for event in events]
        return sum(scores) / max(1, len(scores))

    def _authenticity_risk(self, events: list[NewsEvent]) -> float:
        if not events:
            return 0.0
        risk = 0.0
        for event in events:
            source_score = self._source_score(event.source)
            title = event.title.upper()
            unverified = any(token in title for token in ("BREAKING", "RUMOR", "UNCONFIRMED", "LEAK", "PANIC"))
            viral = any(token in title for token in ("TWITTER", "X POST", "SOCIAL", "REDDIT"))
            if unverified:
                risk += (1.0 - source_score) * 0.8
            if viral:
                risk += (1.0 - source_score) * 0.6
        return min(1.0, risk / max(1, len(events)))

    def _sentiment_extreme(
        self,
        symbol: str,
        events: list[NewsEvent],
        now_utc: datetime,
    ) -> tuple[float, str]:
        if not events:
            return 0.0, "neutral"
        horizon = now_utc.astimezone(UTC) - timedelta(minutes=max(5, self.bias_lookback_minutes // 2))
        scoped = [event for event in events if event.timestamp >= horizon]
        if not scoped:
            return 0.0, "neutral"
        bias_direction, bias_confidence, _ = self._derive_symbol_bias(symbol, scoped, now_utc)
        same_direction = 0
        directional = 0
        for event in scoped:
            score = self._event_direction_score(event.title)
            if abs(score) < 0.01:
                continue
            directional += 1
            if (score > 0 and bias_direction == "bullish") or (score < 0 and bias_direction == "bearish"):
                same_direction += 1
        if directional <= 0 or bias_direction == "neutral":
            return 0.0, "neutral"
        herd_ratio = same_direction / max(1, directional)
        extreme = min(1.0, herd_ratio * max(0.0, bias_confidence))
        if extreme < 0.65:
            return extreme, "neutral"
        return extreme, bias_direction

    @staticmethod
    def _source_score(source: str) -> float:
        name = str(source or "").strip().upper()
        if not name:
            return 0.35
        high = ("BLOOMBERG", "REUTERS", "WSJ", "FINANCIAL TIMES", "CNBC", "AP", "BBC", "FOREX FACTORY", "FINNHUB")
        medium = ("INVESTING", "MARKETWATCH", "YAHOO", "NEWSAPI")
        low = ("TWITTER", "X", "REDDIT", "TELEGRAM", "DISCORD")
        if any(token in name for token in high):
            return 0.95
        if any(token in name for token in medium):
            return 0.70
        if any(token in name for token in low):
            return 0.25
        return 0.45

    @staticmethod
    def _event_direction_score(title: str) -> float:
        text = title.upper()
        positive = (
            "DOVISH",
            "RATE CUT",
            "COOLING INFLATION",
            "SOFT CPI",
            "WEAK USD",
            "RISK-ON",
            "STIMULUS",
        )
        negative = (
            "HAWKISH",
            "RATE HIKE",
            "HOT INFLATION",
            "STRONG USD",
            "RISK-OFF",
            "RECESSION",
            "TIGHTENING",
        )
        score = 0.0
        for token in positive:
            if token in text:
                score += 1.0
        for token in negative:
            if token in text:
                score -= 1.0
        return score

    @staticmethod
    def _oil_supply_direction_score(title: str) -> float:
        text = str(title or "").upper()
        bullish = (
            "CRUDE DRAW",
            "INVENTORY DRAW",
            "SUPPLY CUT",
            "OUTPUT CUT",
            "PIPELINE DISRUPTION",
            "PRODUCTION OUTAGE",
            "SANCTIONS ON OIL",
            "REFINERY OUTAGE",
            "OPEC CUT",
        )
        bearish = (
            "CRUDE BUILD",
            "INVENTORY BUILD",
            "SUPPLY SURGE",
            "OUTPUT HIKE",
            "PRODUCTION INCREASE",
            "RELEASE FROM RESERVES",
            "OPEC BOOST",
            "DEMAND WEAKENS",
        )
        score = 0.0
        for token in bullish:
            if token in text:
                score += 1.0
        for token in bearish:
            if token in text:
                score -= 1.0
        return score

    @staticmethod
    def _geopolitical_risk_score(title: str) -> float:
        text = str(title or "").upper()
        tokens = (
            "WAR",
            "MISSILE",
            "AIRSTRIKE",
            "SANCTION",
            "INVASION",
            "MILITARY",
            "CONFLICT",
            "RED SEA",
            "STRAIT OF HORMUZ",
            "GEOPOLITICAL",
        )
        return 1.0 if any(token in text for token in tokens) else 0.0

    @staticmethod
    def _symbol_bias_score(symbol: str, currency_scores: dict[str, float], risk_score: float) -> float:
        normalized = normalize_symbol_key(symbol)
        asset_class = symbol_asset_class(normalized)
        usd_score = currency_scores.get("USD", 0.0)
        eur_score = currency_scores.get("EUR", 0.0)
        gbp_score = currency_scores.get("GBP", 0.0)
        jpy_score = currency_scores.get("JPY", 0.0)
        aud_score = currency_scores.get("AUD", 0.0)
        nzd_score = currency_scores.get("NZD", 0.0)
        cad_score = currency_scores.get("CAD", 0.0)
        oil_score = currency_scores.get("OIL", 0.0)
        if normalized == "XAUUSD":
            return (-usd_score * 0.75) + (risk_score * 0.30)
        if normalized == "XAGUSD":
            return (-usd_score * 0.60) + (risk_score * 0.18)
        if asset_class == "crypto":
            return (-usd_score * 0.4) + (risk_score * 0.6)
        if normalized == "USOIL":
            return (oil_score * 0.85) - (usd_score * 0.10) + (risk_score * 0.15)
        if normalized == "NAS100":
            return (risk_score * 0.55) - (usd_score * 0.18)
        if asset_class == "equity":
            return (risk_score * 0.48) - (usd_score * 0.16)
        if normalized == "EURUSD":
            return eur_score - usd_score
        if normalized == "GBPUSD":
            return gbp_score - usd_score
        if normalized == "USDJPY":
            return usd_score - jpy_score
        if normalized == "AUDJPY":
            return aud_score - jpy_score + (risk_score * 0.12)
        if normalized == "NZDJPY":
            return nzd_score - jpy_score + (risk_score * 0.12)
        if normalized == "AUDNZD":
            return aud_score - nzd_score + (risk_score * 0.05)
        if normalized == "USDCAD":
            return usd_score - cad_score
        if len(normalized) == 6 and asset_class == "forex":
            return currency_scores.get(normalized[:3], 0.0) - currency_scores.get(normalized[3:6], 0.0)
        return -usd_score

    @staticmethod
    def _categorize_event(*, title: str, source: str, currency: str, description: str = "") -> str:
        text = f"{title} {description}".upper()
        if any(token in text for token in ("WAR", "MISSILE", "SANCTION", "INVASION", "AIRSTRIKE", "CONFLICT", "GEOPOLITICAL", "RED SEA", "HORMUZ")):
            return "geopolitics"
        if any(token in text for token in ("FOMC", "RATE CUT", "RATE HIKE", "CENTRAL BANK", "FED", "ECB", "BOE", "BOJ", "RBA", "RBNZ", "SNB", "BOC", "PBOC")):
            return "monetary_policy"
        if any(token in text for token in ("CPI", "PCE", "INFLATION", "PPI", "PRICES")):
            return "inflation"
        if any(token in text for token in ("NFP", "PAYROLL", "UNEMPLOYMENT", "JOBS", "LABOR", "WAGES")):
            return "labor"
        if any(token in text for token in ("GDP", "PMI", "RETAIL SALES", "GROWTH", "RECESSION", "MANUFACTURING")):
            return "growth"
        if any(token in text for token in ("CRUDE", "OIL", "OPEC", "INVENTORY", "WTI", "BRENT", "SUPPLY CUT", "OUTPUT CUT")):
            return "energy_supply"
        if any(token in text for token in ("GOLD", "BULLION", "SILVER", "COPPER", "METALS")):
            return "commodity_metals"
        if any(token in text for token in ("EARNINGS", "GUIDANCE", "REVENUE", "EPS", "NVDA", "NVIDIA", "AAPL", "APPLE", "MAGNIFICENT 7", "AI CHIP")):
            return "equity_earnings"
        if any(token in text for token in ("BITCOIN", "BTC", "ETH", "CRYPTO", "STABLECOIN", "ETF")):
            return "crypto_market"
        if any(token in text for token in ("NASDAQ", "S&P 500", "DOW", "INDEX", "EQUITIES", "STOCKS", "FUTURES")):
            return "equity_index"
        if any(token in text for token in ("RISK-ON", "RISK OFF", "RISK-OFF", "SAFE HAVEN", "DOLLAR", "YEN", "STERLING", "EURO", "AUSSIE", "KIWI")):
            return "risk_sentiment"
        if str(currency or "").upper() not in {"", "ALL"}:
            return "fx_macro"
        if "FINNHUB" in str(source or "").upper():
            return "calendar_event"
        return "general_macro"

    def _resolve_api_key(self, *, api_key_env: str | None = None, configured_key: str | None = None) -> str:
        env_name = str(api_key_env or self.api_key_env or "").strip()
        env_value = os.getenv(env_name, "").strip() if env_name else ""
        if env_value:
            return env_value
        return str(configured_key if configured_key is not None else self.api_key).strip()

    @staticmethod
    def _currency_from_country(value: str) -> str:
        country = str(value or "").strip().upper()
        mapping = {
            "UNITED STATES": "USD",
            "US": "USD",
            "USD": "USD",
            "EURO AREA": "EUR",
            "EU": "EUR",
            "EUR": "EUR",
            "UNITED KINGDOM": "GBP",
            "UK": "GBP",
            "GBP": "GBP",
            "JAPAN": "JPY",
            "JP": "JPY",
            "JPY": "JPY",
            "AUSTRALIA": "AUD",
            "AU": "AUD",
            "AUD": "AUD",
            "NEW ZEALAND": "NZD",
            "NZ": "NZD",
            "NZD": "NZD",
            "CANADA": "CAD",
            "CA": "CAD",
            "CAD": "CAD",
            "SWITZERLAND": "CHF",
            "CH": "CHF",
            "CHF": "CHF",
            "CHINA": "CNY",
            "CN": "CNY",
            "CNY": "CNY",
        }
        return mapping.get(country, "ALL")

    def _load_cache(self) -> dict[str, Any] | None:
        if not self.cache_path.exists():
            return None
        try:
            return json.loads(self.cache_path.read_text())
        except Exception:
            return None

    def _save_cache(self, events: list[NewsEvent], source: str, now_utc: datetime, api_ok: bool) -> None:
        ensure_parent(self.cache_path)
        payload = {
            "fetched_at": now_utc.isoformat(),
            "expires_at": (now_utc + timedelta(seconds=self.cache_ttl_seconds)).isoformat(),
            "source": source,
            "api_ok": api_ok,
            "events": [
                {
                    **asdict(event),
                    "timestamp": event.timestamp.isoformat(),
                }
                for event in events
            ],
        }
        self.cache_path.write_text(json.dumps(payload, indent=2))

    def _events_from_payload(self, payload: list[dict[str, Any]], source: str) -> tuple[list[NewsEvent], str]:
        events: list[NewsEvent] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            parsed_time = self._parse_timestamp(item.get("timestamp"))
            if parsed_time is None:
                continue
            events.append(
                NewsEvent(
                    event_id=str(item.get("event_id") or item.get("id") or f"cached-{parsed_time.isoformat()}"),
                    timestamp=parsed_time,
                    title=str(item.get("title", "event")),
                    currency=str(item.get("currency", "ALL")).upper(),
                    impact=self._normalize_impact(item.get("impact")),
                    source=str(item.get("source", source)),
                    category=str(item.get("category") or "general_macro"),
                    is_major_risk=bool(item.get("is_major_risk", False)),
                )
            )
        return events, source

    def _log(self, symbol: str, decision: NewsDecision) -> None:
        prefix = "NEWS OK" if decision.safe else "NEWS BLOCK"
        message = (
            f"{prefix}: {symbol.upper()} state={decision.state} reason={decision.reason} "
            f"source={decision.source_used or decision.source} fallback={int(bool(decision.fallback_used))} "
            f"decision_conf={decision.decision_confidence:.2f} bias={decision.bias_direction}:{decision.bias_confidence:.2f} "
            f"src={decision.source_confidence:.2f} auth={decision.authenticity_risk:.2f}"
        )
        signature = (
            f"{decision.safe}:{decision.state}:{decision.reason}:{decision.bias_direction}:{round(decision.bias_confidence, 2)}:"
            f"{round(decision.source_confidence, 2)}:{round(decision.authenticity_risk, 2)}:{round(decision.decision_confidence, 2)}"
        )
        now = datetime.now(tz=UTC)
        last_signature = self._last_log_state.get(symbol.upper())
        last_time = self._last_log_time.get(symbol.upper())
        if last_signature == signature and last_time is not None:
            if (now - last_time).total_seconds() < max(0, self.log_refresh_seconds):
                return
        self._last_log_state[symbol.upper()] = signature
        self._last_log_time[symbol.upper()] = now
        logger = self.logger
        if logger is not None:
            if decision.safe and hasattr(logger, "info"):
                logger.info(message)
                return
            if (not decision.safe) and hasattr(logger, "warning"):
                logger.warning(message)
                return
        print(message)

    def _fetch_url(self, url: str) -> str:
        attempts = max(1, int(self.http_retries) + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                if self.http_get is not None:
                    return self.http_get(url)
                return self._default_http_get(
                    url,
                    timeout_seconds=max(1.0, float(self.http_timeout_seconds)),
                    user_agent=self.user_agent,
                )
            except HTTPError as exc:
                if exc.code in {401, 403}:
                    raise RuntimeError(f"provider_auth_failed:{exc.code}") from exc
                last_error = RuntimeError(f"provider_http_error:{exc.code}")
                if attempt >= attempts - 1:
                    break
                time.sleep(min(1.5, 0.35 * (attempt + 1)))
            except (URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    break
                time.sleep(min(1.5, 0.35 * (attempt + 1)))
            except Exception as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    break
                time.sleep(min(1.5, 0.35 * (attempt + 1)))
        raise RuntimeError(str(last_error or "news_http_failed"))

    @staticmethod
    def _default_http_get(
        url: str,
        *,
        timeout_seconds: float = 3.0,
        user_agent: str = "Mozilla/5.0 (ApexBot News)",
    ) -> str:
        request = Request(url, headers={"User-Agent": user_agent})
        with urlopen(request, timeout=max(1.0, float(timeout_seconds))) as response:  # nosec B310
            return response.read().decode("utf-8")

    @staticmethod
    def _normalize_impact(value: Any) -> str:
        text = str(value or "LOW").strip().upper()
        if text in {"3", "HIGH", "RED", "HIGH IMPACT"}:
            return "HIGH"
        if text in {"2", "MED", "MEDIUM", "ORANGE", "MEDIUM IMPACT"}:
            return "MED"
        return "LOW"

    @staticmethod
    def _looks_high_impact_title(title: str) -> bool:
        headline = title.upper()
        tokens = ("FOMC", "FED", "RATE", "CPI", "PCE", "NFP", "PAYROLL", "GDP", "ECB", "BOE", "BOJ", "INFLATION")
        return any(token in headline for token in tokens)

    @staticmethod
    def _infer_newsapi_currency(title: str) -> str:
        headline = title.upper()
        if "RBA" in headline or "AUD" in headline or "AUSSIE" in headline or "AUSTRALIAN DOLLAR" in headline:
            return "AUD"
        if "RBNZ" in headline or "NZD" in headline or "KIWI" in headline:
            return "NZD"
        if "BOC" in headline or "CAD" in headline or "LOONIE" in headline:
            return "CAD"
        if "SNB" in headline or "CHF" in headline or "FRANC" in headline:
            return "CHF"
        if "PBOC" in headline or "YUAN" in headline or "CNY" in headline:
            return "CNY"
        if "ECB" in headline or "EURO" in headline or "EUR" in headline:
            return "EUR"
        if "BOE" in headline or "STERLING" in headline or "POUND" in headline or "GBP" in headline:
            return "GBP"
        if "BOJ" in headline or "YEN" in headline or "JPY" in headline:
            return "JPY"
        if any(token in headline for token in ("FED", "FOMC", "USD", "DOLLAR", "CPI", "PCE", "NFP", "PAYROLL", "GDP")):
            return "USD"
        return "ALL"

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=UTC)
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        for candidate in (normalized, normalized.replace(" ", "T")):
            try:
                parsed = datetime.fromisoformat(candidate)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=UTC)
                return parsed.astimezone(UTC)
            except ValueError:
                continue
        try:
            parsed = parsedate_to_datetime(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except Exception:
            return None
