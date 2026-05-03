from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import html
import json
import os

from src.hyperliquid_lab.config import HyperliquidLabConfig


class TelegramBotError(RuntimeError):
    pass


@dataclass(frozen=True)
class TelegramRuntimeConfig:
    enabled: bool = False
    bot_username: str = ""
    token_env: str = "TELEGRAM_BOT_TOKEN"
    chat_id_env: str = "TELEGRAM_CHAT_ID"
    parse_mode: str = "HTML"
    notify_on: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class TelegramBotClient:
    token: str
    api_base_url: str = "https://api.telegram.org"
    timeout_seconds: float = 10.0

    def get_me(self) -> dict[str, Any]:
        return self._request("getMe")

    def get_updates(self, *, offset: int | None = None, timeout: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": int(timeout), "limit": int(limit)}
        if offset is not None:
            payload["offset"] = int(offset)
        result = self._request("getUpdates", payload)
        if not isinstance(result, list):
            raise TelegramBotError("getUpdates returned an unexpected payload")
        return result

    def send_message(self, chat_id: int | str, text: str, *, parse_mode: str = "HTML", disable_web_page_preview: bool = True) -> dict[str, Any]:
        return self._request(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": bool(disable_web_page_preview),
            },
        )

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text[:200]
        return self._request("answerCallbackQuery", payload)

    def discover_chat_ids(self) -> list[int]:
        chat_ids: list[int] = []
        seen: set[int] = set()
        for update in self.get_updates(timeout=0, limit=100):
            chat = _chat_from_update(update)
            if not chat:
                continue
            chat_id = int(chat["id"])
            if chat_id not in seen:
                seen.add(chat_id)
                chat_ids.append(chat_id)
        return chat_ids

    def _request(self, method: str, payload: Mapping[str, Any] | None = None) -> Any:
        if not self.token or ":" not in self.token:
            raise TelegramBotError("Telegram bot token is missing or malformed")
        url = f"{self.api_base_url.rstrip('/')}/bot{self.token}/{method}"
        body = json.dumps(dict(payload or {})).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=float(self.timeout_seconds)) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TelegramBotError(_safe_telegram_error(detail)) from exc
        except URLError as exc:
            raise TelegramBotError(f"Telegram request failed: {exc.reason}") from exc
        payload = json.loads(raw)
        if not bool(payload.get("ok")):
            raise TelegramBotError(_safe_telegram_error(str(payload.get("description", "request failed"))))
        return payload.get("result")


@dataclass
class TelegramNotifier:
    client: TelegramBotClient
    chat_id: int | str
    parse_mode: str = "HTML"

    def notify_event(self, title: str, payload: Mapping[str, Any], *, severity: str = "INFO") -> dict[str, Any]:
        text = format_event_message(title, payload, severity=severity)
        return self.client.send_message(self.chat_id, text, parse_mode=self.parse_mode)

    def send_overview(self, overview: Mapping[str, Any]) -> dict[str, Any]:
        return self.client.send_message(self.chat_id, format_overview_message(overview), parse_mode=self.parse_mode)

    def send_test(self, overview: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = {"status": "telegram_operational", "scope": "hyperliquid_lab_research_paper_only"}
        if overview:
            payload["overview_sections"] = ", ".join(sorted(overview.keys()))
        return self.notify_event("Nexus Vantage Telegram test", payload, severity="TEST")


@dataclass
class TelegramCommandRouter:
    notifier_factory: Callable[[int | str], TelegramNotifier]
    overview_provider: Callable[[], Mapping[str, Any]]

    def handle_update(self, update: Mapping[str, Any]) -> bool:
        message = update.get("message") if isinstance(update, Mapping) else None
        if not isinstance(message, Mapping):
            return False
        chat = message.get("chat")
        if not isinstance(chat, Mapping) or "id" not in chat:
            return False
        text = str(message.get("text") or "").strip()
        if not text.startswith("/"):
            return False
        command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        notifier = self.notifier_factory(chat["id"])
        if command in {"/start", "/help"}:
            notifier.notify_event(
                "Nexus Vantage commands",
                {
                    "/overview": "Full research server overview",
                    "/status": "Condensed health snapshot",
                    "/risk": "Risk and circuit-breaker snapshot",
                    "/paper": "Paper-trading state",
                    "/ping": "Connectivity test",
                },
                severity="INFO",
            )
            return True
        if command in {"/overview", "/status"}:
            notifier.send_overview(self.overview_provider())
            return True
        if command == "/risk":
            overview = self.overview_provider()
            notifier.notify_event("Risk overview", overview.get("risk", {}), severity="INFO")
            return True
        if command == "/paper":
            overview = self.overview_provider()
            notifier.notify_event("Paper overview", overview.get("paper", {}), severity="INFO")
            return True
        if command == "/ping":
            notifier.notify_event("Pong", {"status": "ok"}, severity="INFO")
            return True
        return False


def telegram_config_from_lab_config(config: HyperliquidLabConfig) -> TelegramRuntimeConfig:
    raw = config.raw.get("telegram", {})
    if not isinstance(raw, dict):
        raw = {}
    notify_on = raw.get("notify_on", [])
    return TelegramRuntimeConfig(
        enabled=bool(raw.get("enabled", False)),
        bot_username=str(raw.get("bot_username", "")),
        token_env=str(raw.get("token_env", "TELEGRAM_BOT_TOKEN")),
        chat_id_env=str(raw.get("chat_id_env", "TELEGRAM_CHAT_ID")),
        parse_mode=str(raw.get("parse_mode", "HTML")),
        notify_on=tuple(str(item) for item in notify_on if str(item).strip()),
    )


def client_from_env(runtime_config: TelegramRuntimeConfig) -> TelegramBotClient:
    token = os.getenv(runtime_config.token_env, "").strip()
    if not token:
        raise TelegramBotError(f"Missing {runtime_config.token_env}; set it outside YAML")
    return TelegramBotClient(token=token)


def notifier_from_env(runtime_config: TelegramRuntimeConfig) -> TelegramNotifier:
    client = client_from_env(runtime_config)
    chat_id = os.getenv(runtime_config.chat_id_env, "").strip()
    if not chat_id:
        discovered = client.discover_chat_ids()
        if not discovered:
            raise TelegramBotError(f"Missing {runtime_config.chat_id_env}; send /start to the bot, then run chat discovery")
        chat_id = str(discovered[-1])
    return TelegramNotifier(client=client, chat_id=chat_id, parse_mode=runtime_config.parse_mode)


def build_lab_overview(config: HyperliquidLabConfig, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    overview: dict[str, Any] = {
        "venue": {
            "name": "hyperliquid",
            "market_type": "perps",
            "mode": str(config.raw.get("mode", "research")),
            "assets": ", ".join(config.assets),
            "native_intervals": ", ".join(config.native_intervals),
        },
        "storage": {"root": str(config.storage_root)},
        "fees": {"maker": config.maker_fee_rate, "taker": config.taker_fee_rate},
        "simulator": {
            "max_slippage_bps": config.max_slippage_bps,
            "max_order_book_levels": config.max_order_book_levels,
            "min_fill_ratio": config.min_fill_ratio,
            "stale_book_timeout_seconds": config.stale_book_timeout_seconds,
        },
        "risk": dict(config.raw.get("risk", {})) if isinstance(config.raw.get("risk", {}), dict) else {},
        "paper": dict(config.raw.get("paper", {})) if isinstance(config.raw.get("paper", {}), dict) else {},
        "telegram": {
            "enabled": telegram_config_from_lab_config(config).enabled,
            "bot_username": telegram_config_from_lab_config(config).bot_username,
            "token_source": telegram_config_from_lab_config(config).token_env,
            "chat_id_source": telegram_config_from_lab_config(config).chat_id_env,
        },
    }
    if extra:
        overview.update(dict(extra))
    return overview


def format_event_message(title: str, payload: Mapping[str, Any], *, severity: str) -> str:
    lines = [f"<b>{html.escape(severity.upper())}: {html.escape(title)}</b>"]
    lines.extend(_format_mapping(payload))
    return "\n".join(lines)[:3900]


def format_overview_message(overview: Mapping[str, Any]) -> str:
    lines = ["<b>Nexus Vantage server overview</b>"]
    for section, payload in overview.items():
        lines.append(f"\n<b>{html.escape(str(section).upper())}</b>")
        if isinstance(payload, Mapping):
            lines.extend(_format_mapping(payload))
        else:
            lines.append(f"- {html.escape(str(payload))}")
    return "\n".join(lines)[:3900]


def _format_mapping(payload: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, Mapping):
            lines.append(f"- <b>{html.escape(str(key))}</b>:")
            for child_key, child_value in value.items():
                lines.append(f"  - {html.escape(str(child_key))}: <code>{html.escape(str(child_value))}</code>")
        else:
            lines.append(f"- {html.escape(str(key))}: <code>{html.escape(str(value))}</code>")
    return lines


def _chat_from_update(update: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("message", "edited_message", "channel_post", "callback_query"):
        value = update.get(key)
        if not isinstance(value, Mapping):
            continue
        if key == "callback_query":
            message = value.get("message")
            if isinstance(message, Mapping) and isinstance(message.get("chat"), Mapping):
                return message["chat"]
            continue
        chat = value.get("chat")
        if isinstance(chat, Mapping):
            return chat
    return None


def _safe_telegram_error(detail: str) -> str:
    text = str(detail)
    if "bot" in text and "/" in text:
        return "Telegram API request failed"
    return text[:500]
