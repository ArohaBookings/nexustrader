from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import html
import json
import os
from pathlib import Path
import re
import socket
import time


SAFE_BRIDGE_ACTIONS = {"pause_trading", "resume_trading", "kill_switch", "refresh_state"}
CONFIRMATION_ACTIONS = {"resume_trading", "kill_switch"}
BLOCKED_INTENTS = re.compile(
    r"\b("
    r"buy|sell|long|short|market order|place order|open position|close position|"
    r"increase leverage|raise leverage|increase risk|ignore risk|bypass|yolo|all in|"
    r"change parameters|optimi[sz]e live|disable stop|remove stop"
    r")\b",
    re.IGNORECASE,
)


class ApexTelegramError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApexTelegramConfig:
    enabled: bool = False
    bot_username: str = "Nexus_vantage_trader_bot"
    token_env: str = "TELEGRAM_BOT_TOKEN"
    owner_chat_id_env: str = "TELEGRAM_CHAT_ID"
    webhook_secret_env: str = "TELEGRAM_WEBHOOK_SECRET"
    parse_mode: str = "HTML"
    ai_enabled: bool = True
    allow_controls: bool = True
    test_message_on_startup: bool = False
    confirmation_ttl_seconds: int = 300


@dataclass(frozen=True)
class TelegramMessage:
    chat_id: str
    user_id: str
    username: str
    text: str
    update_id: int = 0


@dataclass(frozen=True)
class TelegramCommandResult:
    handled: bool
    chat_id: str = ""
    user_id: str = ""
    text: str = ""
    action: str = ""
    command_id: str = ""
    confirmation_required: bool = False
    confirm_command_id: str = ""
    parse_mode: str = "HTML"


@dataclass
class TelegramBotClient:
    token: str
    api_base_url: str = "https://api.telegram.org"
    timeout_seconds: float = 10.0

    def get_me(self) -> dict[str, Any]:
        result = self._request("getMe")
        return result if isinstance(result, dict) else {}

    def get_updates(self, *, timeout: int = 0, limit: int = 100, offset: int | None = None) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": int(timeout), "limit": int(limit)}
        if offset is not None:
            payload["offset"] = int(offset)
        transport_timeout = max(float(self.timeout_seconds), float(max(0, int(timeout))) + 10.0)
        result = self._request("getUpdates", payload, timeout_seconds=transport_timeout)
        if not isinstance(result, list):
            raise ApexTelegramError("getUpdates returned unexpected payload")
        return [dict(item) for item in result if isinstance(item, dict)]

    def delete_webhook(self, *, drop_pending_updates: bool = False) -> bool:
        result = self._request("deleteWebhook", {"drop_pending_updates": bool(drop_pending_updates)})
        return bool(result)

    def set_my_commands(self, commands: list[Mapping[str, str]]) -> bool:
        result = self._request("setMyCommands", {"commands": [dict(item) for item in commands]})
        return bool(result)

    def discover_chat_ids(self) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        for update in self.get_updates(timeout=0, limit=100):
            message = extract_telegram_message(update)
            if message and message.chat_id not in seen:
                seen.add(message.chat_id)
                found.append(message.chat_id)
        return found

    def send_message(
        self,
        chat_id: str | int,
        text: str,
        *,
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
    ) -> dict[str, Any]:
        result = self._request(
            "sendMessage",
            {
                "chat_id": str(chat_id),
                "text": str(text)[:3900],
                "parse_mode": parse_mode,
                "disable_web_page_preview": bool(disable_web_page_preview),
            },
        )
        return result if isinstance(result, dict) else {}

    def _request(
        self,
        method: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> Any:
        if not self.token or ":" not in self.token:
            raise ApexTelegramError("Telegram token is missing or malformed")
        body = json.dumps(dict(payload or {})).encode("utf-8")
        request = Request(
            f"{self.api_base_url.rstrip('/')}/bot{self.token}/{method}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            request_timeout = max(1.0, float(self.timeout_seconds if timeout_seconds is None else timeout_seconds))
            with urlopen(request, timeout=request_timeout) as response:  # nosec B310
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ApexTelegramError(_safe_telegram_error(detail)) from exc
        except socket.timeout as exc:
            raise ApexTelegramError("Telegram request timed out") from exc
        except URLError as exc:
            raise ApexTelegramError(f"Telegram request failed: {exc.reason}") from exc
        parsed = json.loads(raw)
        if not bool(parsed.get("ok")):
            raise ApexTelegramError(_safe_telegram_error(str(parsed.get("description") or "request_failed")))
        return parsed.get("result")


class ApexTelegramResponder:
    def __init__(
        self,
        *,
        config: ApexTelegramConfig,
        ai_explainer: Callable[[str, Mapping[str, Any]], tuple[str | None, str | None]] | None = None,
    ) -> None:
        self.config = config
        self.ai_explainer = ai_explainer

    def handle_update(self, update: Mapping[str, Any], dashboard_data: Mapping[str, Any]) -> TelegramCommandResult:
        message = extract_telegram_message(update)
        if message is None:
            return TelegramCommandResult(handled=False)
        return self.handle_text(message.text, dashboard_data, chat_id=message.chat_id, user_id=message.user_id)

    def handle_text(
        self,
        text: str,
        dashboard_data: Mapping[str, Any],
        *,
        chat_id: str = "",
        user_id: str = "",
    ) -> TelegramCommandResult:
        raw = str(text or "").strip()
        lower = raw.lower()
        if not raw:
            return TelegramCommandResult(True, chat_id, user_id, _help_text(), parse_mode=self.config.parse_mode)
        if lower.startswith("/confirm"):
            parts = raw.split()
            command_id = parts[1].strip() if len(parts) > 1 else ""
            reply = "Send <code>/confirm COMMAND_ID</code>." if not command_id else ""
            return TelegramCommandResult(
                handled=True,
                chat_id=chat_id,
                user_id=user_id,
                text=reply,
                confirm_command_id=command_id,
                parse_mode=self.config.parse_mode,
            )
        if lower in {"/start", "/help", "help"}:
            return self._result(chat_id, user_id, _help_text())
        if lower in {"/status", "status"}:
            return self._result(chat_id, user_id, format_status(dashboard_data))
        if lower in {"/funded", "funded"} or re.search(r"\bfunded\b.*\b(pass|target|drawdown|account|status)\b", raw, re.I):
            return self._result(chat_id, user_id, format_funded(dashboard_data))
        if lower in {"/trajectory", "trajectory"} or re.search(r"\b(trajectory|100k|100k path|forecast)\b", raw, re.I):
            return self._result(chat_id, user_id, format_trajectory(dashboard_data))
        if lower in {"/losses", "losses", "review losses"} or re.search(r"\b(losses|loss review|why losing|losing too much)\b", raw, re.I):
            return self._result(chat_id, user_id, format_losses(dashboard_data))
        if lower in {"/blockers", "blockers"} or re.search(r"\b(blockers|blocked|why blocked|not trading)\b", raw, re.I):
            return self._result(chat_id, user_id, format_blockers(dashboard_data))
        if lower in {"/risk", "risk"}:
            return self._result(chat_id, user_id, format_risk(dashboard_data))
        if lower in {"/trades", "trades"}:
            return self._result(chat_id, user_id, format_trades(dashboard_data))
        if lower in {"/apex", "/intel", "apex", "intel"} or re.search(r"\b(thinking|edge|scale|100k|intelligence|apex|repair|self[- ]?heal)\b", raw, re.I):
            return self._result(chat_id, user_id, format_apex(dashboard_data))
        if re.search(r"\b(increase|more|raise|boost).*\b(frequency|trades|entries)\b|\bfrequency\b.*\b(xau|btc|gold)\b", raw, re.I):
            return self._result(chat_id, user_id, format_frequency_policy(dashboard_data))
        command_action = {
            "/pause": "pause_trading",
            "/resume": "resume_trading",
            "/kill": "kill_switch",
            "/refresh": "refresh_state",
        }.get(lower)
        if command_action:
            if not bool(self.config.allow_controls):
                return self._result(chat_id, user_id, "Telegram controls are disabled by config.")
            command_id = _command_id()
            confirmation_required = command_action in CONFIRMATION_ACTIONS
            if confirmation_required:
                return TelegramCommandResult(
                    handled=True,
                    chat_id=chat_id,
                    user_id=user_id,
                    text=(
                        f"Confirmation required for <code>{html.escape(command_action)}</code>.\n"
                        f"Send <code>/confirm {html.escape(command_id)}</code> to execute."
                    ),
                    action=command_action,
                    command_id=command_id,
                    confirmation_required=True,
                    parse_mode=self.config.parse_mode,
                )
            return TelegramCommandResult(
                handled=True,
                chat_id=chat_id,
                user_id=user_id,
                text=f"Executing <code>{html.escape(command_action)}</code>.",
                action=command_action,
                command_id=command_id,
                parse_mode=self.config.parse_mode,
            )
        if BLOCKED_INTENTS.search(raw):
            return self._result(
                chat_id,
                user_id,
                "Blocked: Telegram cannot place trades, increase risk, bypass funded rails, or change live strategy parameters. Use /status, /funded, /risk, /trades, /apex, /pause, /refresh, /resume, or /kill.",
            )
        ai_text = self._ai_reply(raw, dashboard_data)
        return self._result(chat_id, user_id, ai_text)

    def _ai_reply(self, question: str, dashboard_data: Mapping[str, Any]) -> str:
        if bool(self.config.ai_enabled) and self.ai_explainer is not None:
            try:
                answer, error = self.ai_explainer(question, dashboard_data)
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                answer, error = None, str(exc)
            if answer:
                return _html_lines("<b>APEX AI</b>", [answer])
            if error and error not in {"missing_api_key", "remote_unavailable"}:
                return _html_lines("<b>APEX AI fallback</b>", [f"Remote AI unavailable: {error}", local_explanation(question, dashboard_data)])
        return local_explanation(question, dashboard_data)

    def _result(self, chat_id: str, user_id: str, text: str) -> TelegramCommandResult:
        return TelegramCommandResult(True, chat_id, user_id, text, parse_mode=self.config.parse_mode)


def telegram_config_from_mapping(payload: Mapping[str, Any] | None) -> ApexTelegramConfig:
    raw = dict(payload or {})
    return ApexTelegramConfig(
        enabled=bool(raw.get("enabled", False)),
        bot_username=str(raw.get("bot_username", "Nexus_vantage_trader_bot")),
        token_env=str(raw.get("token_env", "TELEGRAM_BOT_TOKEN")),
        owner_chat_id_env=str(raw.get("owner_chat_id_env", raw.get("chat_id_env", "TELEGRAM_CHAT_ID"))),
        webhook_secret_env=str(raw.get("webhook_secret_env", "TELEGRAM_WEBHOOK_SECRET")),
        parse_mode=str(raw.get("parse_mode", "HTML")),
        ai_enabled=bool(raw.get("ai_enabled", True)),
        allow_controls=bool(raw.get("allow_controls", True)),
        test_message_on_startup=bool(raw.get("test_message_on_startup", False)),
        confirmation_ttl_seconds=max(30, int(_number(raw.get("confirmation_ttl_seconds"), 300.0))),
    )


def client_from_env(config: ApexTelegramConfig) -> TelegramBotClient:
    token = os.getenv(config.token_env, "").strip()
    if not token:
        raise ApexTelegramError(f"Missing {config.token_env}; set the token outside source files")
    return TelegramBotClient(token=token)


def owner_chat_id_from_env(config: ApexTelegramConfig, client: TelegramBotClient | None = None) -> str:
    chat_id = os.getenv(config.owner_chat_id_env, "").strip()
    if chat_id:
        return chat_id
    chat_id = _secret_file_value(config.owner_chat_id_env)
    if chat_id:
        os.environ[config.owner_chat_id_env] = chat_id
        return chat_id
    if client is not None:
        discovered = client.discover_chat_ids()
        if discovered:
            return discovered[-1]
    return ""


def _secret_file_value(key: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(key or "")):
        return ""
    for path in _candidate_secret_paths():
        if not path.exists() or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            raw_key, raw_value = stripped.split("=", 1)
            if raw_key.strip() == key:
                return raw_value.strip().strip('"').strip("'")
    return ""


def _candidate_secret_paths() -> list[Path]:
    cwd = Path.cwd()
    module_root = Path(__file__).resolve().parents[1]
    candidates = [
        cwd / "config" / "secrets.env",
        cwd / "secrets.env",
        module_root / "config" / "secrets.env",
        module_root / "secrets.env",
    ]
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def extract_telegram_message(update: Mapping[str, Any]) -> TelegramMessage | None:
    update_id = int(_number(update.get("update_id"), 0.0))
    message = update.get("message")
    if not isinstance(message, Mapping):
        message = update.get("edited_message")
    if not isinstance(message, Mapping):
        callback = update.get("callback_query")
        if isinstance(callback, Mapping):
            message = callback.get("message")
            data = str(callback.get("data") or "")
        else:
            data = ""
    else:
        data = ""
    if not isinstance(message, Mapping):
        return None
    chat = message.get("chat")
    if not isinstance(chat, Mapping) or chat.get("id") is None:
        return None
    sender = message.get("from") if isinstance(message.get("from"), Mapping) else {}
    text = str(message.get("text") or data or "").strip()
    if not text:
        return None
    return TelegramMessage(
        chat_id=str(chat.get("id")),
        user_id=str(sender.get("id") or ""),
        username=str(sender.get("username") or ""),
        text=text,
        update_id=update_id,
    )


def format_status(dashboard_data: Mapping[str, Any]) -> str:
    summary = _record(dashboard_data.get("summary"))
    apex = _record(dashboard_data.get("institutional_apex"))
    funded = _record(apex.get("funded_mission"))
    mt5 = _record(apex.get("mt5_bridge"))
    return _html_lines(
        "<b>APEX Status</b>",
        [
            f"Readiness: <code>{_esc(apex.get('readiness', 'unknown'))}</code> | Grade: <code>{_number(apex.get('grade_pct'), 0.0):.1f}%</code>",
            f"MT5: <code>{'CONNECTED' if mt5.get('connected') else 'DISCONNECTED'}</code> account <code>{_esc(mt5.get('account', ''))}</code>",
            f"Equity: <code>{_money(_number(summary.get('equity') or mt5.get('equity'), 0.0))}</code> | Free margin: <code>{_money(_number(summary.get('free_margin') or mt5.get('free_margin'), 0.0))}</code>",
            f"Daily state: <code>{_esc(summary.get('current_daily_state', ''))}</code> | Session: <code>{_esc(summary.get('current_session', ''))}</code>",
            f"Funded: <code>{_esc(funded.get('status', 'disabled'))}</code> | Needed: <code>{_money(_number(funded.get('needed_to_pass'), 0.0))}</code>",
            f"Queue: <code>{_number(summary.get('queued_actions_total'), 0.0):.0f}</code> | Open trades: <code>{_number(summary.get('open_positions'), 0.0):.0f}</code>",
        ],
    )


def format_funded(dashboard_data: Mapping[str, Any]) -> str:
    apex = _record(dashboard_data.get("institutional_apex"))
    funded = _record(apex.get("funded_mission"))
    account = _record(funded.get("account"))
    return _html_lines(
        "<b>Funded Mission</b>",
        [
            f"Enabled: <code>{bool(funded.get('enabled'))}</code> | Group: <code>{_esc(funded.get('group', 'custom'))}</code> | Phase: <code>{_esc(funded.get('phase', 'evaluation'))}</code>",
            f"Status: <code>{_esc(funded.get('status', 'disabled'))}</code> | Guard: <code>{_esc(funded.get('guard_reason', ''))}</code>",
            f"MT5 equity: <code>{_money(_number(account.get('equity'), 0.0))}</code> | Start: <code>{_money(_number(account.get('starting_balance'), 0.0))}</code>",
            f"Target equity: <code>{_money(_number(funded.get('target_equity'), 0.0))}</code> | Needed: <code>{_money(_number(funded.get('needed_to_pass'), 0.0))}</code>",
            f"Daily buffer: <code>{_money(_number(funded.get('daily_buffer_usd'), 0.0))}</code> ({_pct(_number(funded.get('daily_buffer_pct'), 0.0))})",
            f"Overall buffer: <code>{_money(_number(funded.get('overall_buffer_usd'), 0.0))}</code> ({_pct(_number(funded.get('overall_buffer_pct'), 0.0))})",
            f"Throttle: <code>{_pct(_number(funded.get('risk_throttle'), 0.0))}</code> | Max risk/trade: <code>{_money(_number(funded.get('max_risk_per_trade_usd'), 0.0))}</code>",
        ],
    )


def format_risk(dashboard_data: Mapping[str, Any]) -> str:
    health = _record(dashboard_data.get("health"))
    summary = _record(dashboard_data.get("summary"))
    apex = _record(dashboard_data.get("institutional_apex"))
    edge = _record(dashboard_data.get("institutional_intelligence"))
    repair = _record(apex.get("self_repair"))
    edge_repair = _record(edge.get("self_repair") or dashboard_data.get("self_repair"))
    if edge_repair:
        repair = edge_repair
    return _html_lines(
        "<b>Risk</b>",
        [
            f"Daily state: <code>{_esc(health.get('current_daily_state') or summary.get('current_daily_state') or '')}</code>",
            f"Reason: <code>{_esc(health.get('current_daily_state_reason') or summary.get('current_daily_state_reason') or '')}</code>",
            f"Open risk: <code>{_pct(_number(health.get('open_risk_pct'), 0.0))}</code> | Daily DD: <code>{_pct(_number(summary.get('daily_dd_pct_live'), 0.0))}</code>",
            f"Repair: <code>{_esc(repair.get('status', 'unknown'))}</code> | Soft blockers: <code>{len(_sequence(repair.get('soft_blockers')))}</code> | Hard rails: <code>{len(_sequence(repair.get('hard_rails')))}</code>",
            "Hard drawdown, funded, stale-data, broker, and kill rails are not auto-overridden.",
        ],
    )


def format_trajectory(dashboard_data: Mapping[str, Any]) -> str:
    trajectory = _record(dashboard_data.get("trajectory_forecast"))
    edge = _record(dashboard_data.get("institutional_intelligence"))
    if not trajectory:
        trajectory = _record(edge.get("trajectory_forecast"))
    if not trajectory:
        return "<b>Trajectory</b>\nNo trajectory forecast is available yet."
    return _html_lines(
        "<b>Trajectory</b>",
        [
            f"Current equity: <code>{_money(_number(trajectory.get('current_equity'), 0.0))}</code>",
            f"Short goal: <code>{_money(_number(trajectory.get('short_goal_equity'), 100000.0))}</code> in <code>{_number(trajectory.get('short_goal_days'), 0.0):.0f}</code> days | On track: <code>{bool(trajectory.get('short_goal_on_track'))}</code>",
            f"Medium goal: <code>{_money(_number(trajectory.get('medium_goal_equity'), 1000000.0))}</code> | On track: <code>{bool(trajectory.get('medium_goal_on_track'))}</code>",
            f"Forecast type: <code>{_esc(trajectory.get('forecast_type', 'speculative_target_path'))}</code>",
            "This forecast is not a sizing input and does not authorize higher risk.",
        ],
    )


def format_losses(dashboard_data: Mapping[str, Any]) -> str:
    health = _record(dashboard_data.get("health"))
    rollout = _record(health.get("current_rollout_stats"))
    overall = _record(rollout.get("overall"))
    edge = _record(dashboard_data.get("institutional_intelligence"))
    live_shadow = _record(dashboard_data.get("live_shadow_gap") or edge.get("live_shadow_gap"))
    priority = [_record(item) for item in _sequence(live_shadow.get("priority_symbols"))]
    return _html_lines(
        "<b>Loss Review</b>",
        [
            f"Rollout trades: <code>{_number(rollout.get('trade_count'), _number(overall.get('trades'), 0.0)):.0f}</code> | Win rate: <code>{_pct(_number(overall.get('win_rate'), 0.0))}</code>",
            f"Expectancy: <code>{_number(overall.get('expectancy_r'), 0.0):.3f}R</code> | Profit factor: <code>{_number(overall.get('profit_factor'), 0.0):.2f}</code>",
            f"Live-shadow status: <code>{_esc(live_shadow.get('status', 'collecting_or_aligned'))}</code> | Max gap: <code>{_pct(_number(live_shadow.get('max_gap_score'), 0.0))}</code>",
            "Priority gaps: "
            + _esc(", ".join(f"{item.get('symbol')}:{item.get('status')}" for item in priority[:4]) or "insufficient samples"),
            "Next safe action: collect more BTCUSD/XAUUSD live-shadow evidence before promoting risk or frequency.",
        ],
    )


def format_blockers(dashboard_data: Mapping[str, Any]) -> str:
    edge = _record(dashboard_data.get("institutional_intelligence"))
    repair = _record(dashboard_data.get("self_repair") or edge.get("self_repair"))
    pipeline = _record(dashboard_data.get("xau_btc_opportunity_pipeline") or edge.get("xau_btc_opportunity_pipeline"))
    priority = [_record(item) for item in _sequence(pipeline.get("priority_symbols"))]
    soft = [_record(item) for item in _sequence(repair.get("soft_blockers"))]
    hard = [_record(item) for item in _sequence(repair.get("hard_rails"))]
    priority_lines = [
        f"{item.get('symbol')}: {item.get('live_gate')} | {item.get('recommended_action')} | blocker={item.get('blocker') or 'none'}"
        for item in priority[:4]
    ]
    return _html_lines(
        "<b>Blockers</b>",
        [
            f"Repair status: <code>{_esc(repair.get('status', 'unknown'))}</code> | Soft: <code>{len(soft)}</code> | Hard: <code>{len(hard)}</code>",
            f"Recommended bridge action: <code>{_esc(repair.get('recommended_bridge_action', 'none'))}</code>",
            "BTC/XAU: " + _esc(" ; ".join(priority_lines) or "no priority telemetry yet"),
            "Hard rails are locked and will not be auto-repaired.",
        ],
    )


def format_frequency_policy(dashboard_data: Mapping[str, Any]) -> str:
    edge = _record(dashboard_data.get("institutional_intelligence"))
    pipeline = _record(dashboard_data.get("xau_btc_opportunity_pipeline") or edge.get("xau_btc_opportunity_pipeline"))
    priority = [_record(item) for item in _sequence(pipeline.get("priority_symbols"))]
    lines = []
    for item in priority[:4]:
        target = _record(item.get("shadow_target_10m"))
        lines.append(
            f"{item.get('symbol')}: shadow target {int(_number(target.get('low'), 0.0))}-{int(_number(target.get('high'), 0.0))}/10m, "
            f"candidates {int(_number(item.get('actual_candidates_last_10m'), 0.0))}, "
            f"live {int(_number(item.get('actual_live_trades_last_10m'), 0.0))}, "
            f"gate {item.get('live_gate')}"
        )
    return _html_lines(
        "<b>Frequency Policy</b>",
        [
            "Live frequency is edge-gated; Telegram cannot force entries or raise risk.",
            *[_esc(line) for line in lines],
            f"Forced live frequency: <code>{bool(pipeline.get('live_frequency_forced'))}</code>",
            "Safe response to under-trading is more shadow sampling and blocker diagnostics first.",
        ],
    )


def format_trades(dashboard_data: Mapping[str, Any]) -> str:
    trades = [_record(item) for item in _sequence(dashboard_data.get("open_trades"))]
    if not trades:
        return "<b>Trades</b>\nNo open trades in the bridge journal."
    lines = []
    for trade in trades[:8]:
        lines.append(
            f"{_esc(trade.get('symbol', '?'))} {_esc(trade.get('side', ''))} "
            f"lot <code>{_number(trade.get('size'), 0.0):.2f}</code> "
            f"PnL <code>{_money(_number(trade.get('current_pnl'), 0.0))}</code> "
            f"R <code>{_number(trade.get('current_r'), 0.0):.2f}</code>"
        )
    return _html_lines("<b>Trades</b>", lines)


def format_apex(dashboard_data: Mapping[str, Any]) -> str:
    apex = _record(dashboard_data.get("institutional_apex"))
    edge = _record(dashboard_data.get("institutional_intelligence"))
    if not apex:
        return "<b>Institutional Apex</b>\nNo Apex snapshot is available yet."
    training = _record(dashboard_data.get("training_bootstrap_status") or edge.get("training_bootstrap_status"))
    promotion = _record(dashboard_data.get("promotion_audit") or edge.get("promotion_audit"))
    pipeline = _record(dashboard_data.get("xau_btc_opportunity_pipeline") or edge.get("xau_btc_opportunity_pipeline"))
    return _html_lines(
        "<b>Institutional Apex</b>",
        [
            f"Readiness: <code>{_esc(apex.get('readiness', 'unknown'))}</code> | Grade: <code>{_number(apex.get('grade_pct'), 0.0):.1f}%</code>",
            _esc(apex.get("summary", "")),
            f"Market mastery: <code>{_pct(_number(_record(apex.get('market_mastery')).get('score'), 0.0))}</code>",
            f"Data fusion: <code>{_pct(_number(_record(apex.get('data_fusion')).get('consensus_score'), 0.0))}</code>",
            f"Anti-overfit: <code>{_esc(promotion.get('reason') or _record(apex.get('anti_overfit')).get('reason', 'unknown'))}</code>",
            f"Self-repair: <code>{_esc(_record(apex.get('self_repair')).get('status', 'unknown'))}</code>",
            f"Scaling: <code>{_esc(_record(apex.get('scaling')).get('aggression', 'unknown'))}</code>",
            f"Training: <code>{_esc(training.get('status', 'unknown'))}</code> | Samples: <code>{_number(training.get('trained_samples'), 0.0):.0f}</code>",
            f"BTC/XAU policy: <code>{_esc(pipeline.get('policy', 'edge_gated'))}</code> | Forced live freq: <code>{bool(pipeline.get('live_frequency_forced'))}</code>",
        ],
    )


def local_explanation(question: str, dashboard_data: Mapping[str, Any]) -> str:
    apex = _record(dashboard_data.get("institutional_apex"))
    edge = _record(dashboard_data.get("institutional_intelligence"))
    summary = _record(dashboard_data.get("summary"))
    symbols = [_record(item) for item in _sequence(dashboard_data.get("symbols"))]
    pipeline = _record(dashboard_data.get("xau_btc_opportunity_pipeline") or edge.get("xau_btc_opportunity_pipeline"))
    priority = [_record(item) for item in _sequence(pipeline.get("priority_symbols"))]
    blockers = [item for item in symbols if str(item.get("blocked_reason") or item.get("primary_block_reason") or "").strip()]
    blocker_summary = "; ".join(
        f"{item.get('symbol')}:{item.get('blocked_reason') or item.get('primary_block_reason')}"
        for item in blockers[:4]
    )
    priority_summary = "; ".join(
        f"{item.get('symbol')} debt={item.get('candidate_debt_10m')} gate={item.get('live_gate')}"
        for item in priority[:2]
    )
    return _html_lines(
        "<b>APEX Operator Answer</b>",
        [
            f"Question: <code>{_esc(question)[:500]}</code>",
            f"Readiness: <code>{_esc(apex.get('readiness', 'unknown'))}</code> | Grade: <code>{_number(apex.get('grade_pct'), 0.0):.1f}%</code>",
            _esc(apex.get("summary", "No institutional snapshot available.")),
            f"Equity: <code>{_money(_number(summary.get('equity'), 0.0))}</code> | Daily state: <code>{_esc(summary.get('current_daily_state', ''))}</code>",
            f"Blocked symbols: <code>{len(blockers)}</code>. Main blockers: <code>{_esc(blocker_summary)}</code>",
            f"BTC/XAU opportunity: <code>{_esc(priority_summary or 'waiting for priority telemetry')}</code>",
            "Telegram scope: explanations plus pause, refresh, resume confirmation, and kill confirmation only.",
        ],
    )


def _help_text() -> str:
    return _html_lines(
        "<b>Nexus Vantage APEX Telegram</b>",
        [
            "/status - MT5, equity, daily state, funded summary",
            "/funded - pass target, daily/overall drawdown buffer, throttle",
            "/trajectory - speculative $100 to $100K target path",
            "/losses - live rollout and live-shadow gap review",
            "/blockers - BTC/XAU blockers and safe repair status",
            "/risk - drawdown, open risk, repair rails",
            "/trades - open trade management snapshot",
            "/apex - institutional intelligence and self-repair summary",
            "/pause - pause new trading",
            "/refresh - refresh bridge/dashboard state",
            "/resume - requires /confirm",
            "/kill - requires /confirm",
            "Normal English questions are answered from live telemetry. Trade placement and risk bypasses are blocked.",
        ],
    )


def _html_lines(title: str, lines: list[str]) -> str:
    return "\n".join([title, *[f"- {line}" for line in lines]])[:3900]


def _command_id() -> str:
    return f"tg_{int(time.time() * 1000)}"


def _safe_telegram_error(message: str) -> str:
    text = str(message or "request_failed")
    return re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot<TOKEN>", text)[:500]


def _record(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else default


def _money(value: float) -> str:
    return f"${float(value):,.2f}"


def _pct(value: float) -> str:
    return f"{float(value) * 100.0:.1f}%"


def _esc(value: Any) -> str:
    return html.escape(str(value or ""))
