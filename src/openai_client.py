from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import os
import socket
import time
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _safe_json_loads(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


@dataclass
class OpenAIClient:
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 6.0
    retry_once: bool = False
    enabled: bool = True
    logger: Any | None = None
    review_background_enabled: bool = True
    review_web_search_enabled: bool = True
    review_poll_interval_seconds: float = 0.75
    review_max_poll_seconds: float = 18.0
    _last_error: str | None = field(default=None, init=False, repr=False)
    _last_mode: str = field(default="local", init=False, repr=False)
    _last_request_attempts: int = field(default=0, init=False, repr=False)
    _last_timeout_budget_ms: int = field(default=0, init=False, repr=False)
    _cooldown_until: float = field(default=0.0, init=False, repr=False)

    def is_configured(self) -> bool:
        return bool(self._api_key())

    def health(self) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": True, "mode": "local", "last_error": "remote_disabled", "model": self.model}
        if not self.is_configured():
            return {"ok": True, "mode": "local", "last_error": "missing_api_key", "model": self.model}
        mode = "remote" if self._last_mode == "remote" else "fallback"
        return {"ok": True, "mode": mode, "last_error": self._last_error, "model": self.model}

    def test_connectivity(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "remote_disabled"
        if not self.is_configured():
            self._set_mode("local", "missing_api_key")
            return False, "missing_api_key"
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 32,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": "Respond with JSON: {\"ok\": true}"},
            ],
        }
        data, error = self._request(payload)
        if error:
            self._set_mode("fallback", error)
            return False, error
        content = self._message_content(data)
        parsed = _safe_json_loads(content)
        if not parsed or not bool(parsed.get("ok", False)):
            self._set_mode("fallback", "invalid_connectivity_response")
            return False, "invalid_connectivity_response"
        self._set_mode("remote", None)
        return True, "ok"

    def score_trade(self, context: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if not self.enabled:
            self._set_mode("local", "remote_disabled")
            return None, "remote_unavailable"
        if not self.is_configured():
            self._set_mode("fallback", "missing_api_key")
            return None, "remote_unavailable"
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 240,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cautious trading committee. "
                        "Return ONLY JSON with this schema: "
                        "{"
                        "\"approve\": bool, "
                        "\"confidence\": number, "
                        "\"direction\": \"LONG\"|\"SHORT\"|\"NONE\", "
                        "\"reasons\": [string], "
                        "\"risk_adjustment\": {"
                        "\"size_multiplier\": number, "
                        "\"tp_r\": number|null, "
                        "\"trail_mode\": \"ATR\"|\"STRUCTURE\"|\"NONE\", "
                        "\"trail_atr_mult\": number|null, "
                        "\"break_even_r\": number|null, "
                        "\"partial_close_r\": number|null"
                        "}"
                        "}"
                    ),
                },
                {"role": "user", "content": json.dumps(context, sort_keys=True)},
            ],
        }
        data, error = self._request(payload)
        if error:
            self._set_mode("fallback", error)
            return None, error
        parsed = _safe_json_loads(self._message_content(data))
        if parsed is None:
            self._set_mode("fallback", "malformed_json_response")
            return None, "malformed_json_response"
        self._set_mode("remote", None)
        return parsed, None

    def trade_plan(self, context: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if not self.enabled:
            self._set_mode("local", "remote_disabled")
            return None, "remote_unavailable"
        if not self.is_configured():
            self._set_mode("fallback", "missing_api_key")
            return None, "remote_unavailable"
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 360,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return ONLY JSON with this exact schema: "
                        "{"
                        "\"decision\":\"TAKE|PASS\","
                        "\"setup_type\":\"scalp|daytrade|grid_manage\","
                        "\"side\":\"BUY|SELL\","
                        "\"sl_points\":number,"
                        "\"tp_points\":number,"
                        "\"rr_target\":number,"
                        "\"confidence\":number,"
                        "\"expected_value_r\":number,"
                        "\"risk_tier\":\"LOW|NORMAL|HIGH\","
                        "\"management_plan\":{"
                        "\"move_sl_to_be_at_r\":number|null,"
                        "\"trail_after_r\":number|null,"
                        "\"trail_method\":\"atr|structure|fixed|none\","
                        "\"trail_value\":number|null,"
                        "\"take_partial_at_r\":number|null,"
                        "\"time_stop_minutes\":number|null,"
                        "\"early_exit_rules\":string"
                        "},"
                        "\"notes\":string"
                        "}"
                    ),
                },
                {"role": "user", "content": json.dumps(context, sort_keys=True)},
            ],
        }
        data, error = self._request(payload)
        if error:
            self._set_mode("fallback", error)
            return None, error
        parsed = _safe_json_loads(self._message_content(data))
        if parsed is None:
            self._set_mode("fallback", "malformed_trade_plan_json")
            return None, "malformed_trade_plan_json"
        self._set_mode("remote", None)
        return parsed, None

    def management_plan(self, context: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if not self.enabled:
            self._set_mode("local", "remote_disabled")
            return None, "remote_unavailable"
        if not self.is_configured():
            self._set_mode("fallback", "missing_api_key")
            return None, "remote_unavailable"
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 240,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return ONLY JSON with schema: "
                        "{"
                        "\"decision\":\"HOLD|MODIFY|CLOSE\","
                        "\"confidence\":number,"
                        "\"management_plan\":{"
                        "\"move_sl_to_be_at_r\":number|null,"
                        "\"trail_after_r\":number|null,"
                        "\"trail_method\":\"atr|structure|fixed|none\","
                        "\"trail_value\":number|null,"
                        "\"take_partial_at_r\":number|null,"
                        "\"time_stop_minutes\":number|null,"
                        "\"early_exit_rules\":string"
                        "},"
                        "\"notes\":string"
                        "}"
                    ),
                },
                {"role": "user", "content": json.dumps(context, sort_keys=True)},
            ],
        }
        data, error = self._request(payload)
        if error:
            self._set_mode("fallback", error)
            return None, error
        parsed = _safe_json_loads(self._message_content(data))
        if parsed is None:
            self._set_mode("fallback", "malformed_management_plan_json")
            return None, "malformed_management_plan_json"
        self._set_mode("remote", None)
        return parsed, None

    def management_advice(self, context: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if not self.enabled:
            self._set_mode("local", "remote_disabled")
            return None, "remote_unavailable"
        if not self.is_configured():
            self._set_mode("fallback", "missing_api_key")
            return None, "remote_unavailable"
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 180,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return ONLY JSON with keys: "
                        "{\"close_now\": bool, "
                        "\"risk_adjustment\": {"
                        "\"trail_mode\": \"ATR\"|\"STRUCTURE\"|\"NONE\", "
                        "\"trail_atr_mult\": number|null, "
                        "\"break_even_r\": number|null, "
                        "\"partial_close_r\": number|null"
                        "}, "
                        "\"reasons\": [string]}"
                    ),
                },
                {"role": "user", "content": json.dumps(context, sort_keys=True)},
            ],
        }
        data, error = self._request(payload)
        if error:
            self._set_mode("fallback", error)
            return None, error
        parsed = _safe_json_loads(self._message_content(data))
        if parsed is None:
            self._set_mode("fallback", "malformed_management_json")
            return None, "malformed_management_json"
        self._set_mode("remote", None)
        return parsed, None

    def offline_review(self, context: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if not self.enabled:
            self._set_mode("local", "remote_disabled")
            return None, "remote_unavailable"
        if not self.is_configured():
            self._set_mode("fallback", "missing_api_key")
            return None, "remote_unavailable"
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_output_tokens": 900,
            "background": bool(self.review_background_enabled),
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You are an offline trading research reviewer. "
                        "You are NEVER in the live execution path. "
                        "Return ONLY valid JSON with schema: "
                        "{"
                        "\"summary\": string,"
                        "\"weak_patterns\": [string],"
                        "\"strategy_ideas\": [string],"
                        "\"next_cycle_focus\": [string],"
                        "\"reentry_watchlist\": [string],"
                        "\"weekly_trade_ideas\": [string],"
                        "\"hybrid_pair_ideas\": ["
                        "{"
                        "\"symbol\": string,"
                        "\"session_focus\": [string],"
                        "\"setup_bias\": string,"
                        "\"direction_bias\": string,"
                        "\"conviction\": number,"
                        "\"aggression_delta\": number,"
                        "\"threshold_delta\": number,"
                        "\"reason\": string"
                        "}"
                        "]"
                        "}"
                    ),
                },
                {"role": "user", "content": json.dumps(context, sort_keys=True)},
            ],
            "tools": ([{"type": "web_search"}] if self.review_web_search_enabled else []),
            "tool_choice": "auto",
            "include": (["web_search_call.action.sources"] if self.review_web_search_enabled else []),
        }
        data, error = self._responses_request(payload)
        if error:
            self._set_mode("fallback", error)
            return None, error
        parsed = _safe_json_loads(self._response_text(data))
        if parsed is None:
            self._set_mode("fallback", "malformed_offline_review_json")
            return None, "malformed_offline_review_json"
        self._set_mode("remote", None)
        return parsed, None

    def operator_explanation(self, question: str, context: dict[str, Any]) -> tuple[str | None, str | None]:
        if not self.enabled:
            self._set_mode("local", "remote_disabled")
            return None, "remote_unavailable"
        if not self.is_configured():
            self._set_mode("fallback", "missing_api_key")
            return None, "missing_api_key"
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 520,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are APEX's funded-account operator assistant. "
                        "Answer from the supplied telemetry only. Be direct and concise. "
                        "Focus on MT5 connectivity, funded pass risk, account protection, blockers, data quality, "
                        "anti-overfit gates, self-repair, and safe scaling caps. "
                        "Never reveal hidden chain-of-thought. Never place trades, increase risk, bypass drawdown rails, "
                        "or change strategy parameters. Allowed operations are only pause, refresh, resume with confirmation, "
                        "and kill switch with confirmation."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": str(question or "")[:1000],
                            "telemetry": context,
                        },
                        sort_keys=True,
                    ),
                },
            ],
        }
        data, error = self._request(payload)
        if error:
            self._set_mode("fallback", error)
            return None, error
        text = self._message_content(data).strip()
        if not text:
            self._set_mode("fallback", "empty_operator_response")
            return None, "empty_operator_response"
        self._set_mode("remote", None)
        return text[:1800], None

    def _responses_request(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        total_budget_seconds = max(2.0, float(self.timeout_seconds), float(self.review_max_poll_seconds))
        self._last_timeout_budget_ms = int(round(total_budget_seconds * 1000.0))
        self._last_request_attempts = 0
        started_at = time.monotonic()
        if started_at < self._cooldown_until:
            return None, "openai_cooldown_active"
        try:
            self._last_request_attempts += 1
            created = self._post_json(
                f"{self.base_url.rstrip('/')}/responses",
                payload,
                timeout_seconds=max(1.0, min(float(self.timeout_seconds), total_budget_seconds)),
            )
            if not isinstance(created, dict):
                return None, "invalid_api_payload"
            status = self._response_status(created)
            if status in {"completed", "failed", "cancelled", "incomplete"}:
                if status != "completed":
                    return None, self._response_error(created)
                self._cooldown_until = 0.0
                return created, None
            response_id = str(created.get("id") or "").strip()
            if not response_id:
                return None, "openai_request_failed"
            poll_deadline = started_at + total_budget_seconds
            while time.monotonic() < poll_deadline:
                time.sleep(max(0.2, float(self.review_poll_interval_seconds)))
                remaining = poll_deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._last_request_attempts += 1
                polled = self._get_json(
                    self._responses_poll_url(response_id),
                    timeout_seconds=max(0.5, min(remaining, float(self.timeout_seconds))),
                )
                status = self._response_status(polled)
                if status == "completed":
                    self._cooldown_until = 0.0
                    return polled, None
                if status in {"failed", "cancelled", "incomplete"}:
                    return None, self._response_error(polled)
            self._arm_cooldown(total_budget_seconds)
            return None, "openai_timeout"
        except (TimeoutError, socket.timeout):
            self._arm_cooldown(total_budget_seconds)
            return None, "openai_timeout"
        except HTTPError as exc:
            message = self._http_error_message(exc)
            if 500 <= int(exc.code) < 600:
                self._arm_cooldown(total_budget_seconds)
            return None, message
        except URLError:
            self._arm_cooldown(total_budget_seconds)
            return None, "openai_network_error"
        except Exception:
            self._arm_cooldown(total_budget_seconds)
            return None, "openai_request_failed"

    def _responses_poll_url(self, response_id: str) -> str:
        include = ["web_search_call.action.sources"] if self.review_web_search_enabled else []
        query = urlencode({"include[]": include}, doseq=True) if include else ""
        base = f"{self.base_url.rstrip('/')}/responses/{response_id}"
        return f"{base}?{query}" if query else base

    def _request(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        attempts = 2 if self.retry_once else 1
        last_error: str | None = None
        total_budget_seconds = max(0.5, float(self.timeout_seconds))
        self._last_timeout_budget_ms = int(round(total_budget_seconds * 1000.0))
        self._last_request_attempts = 0
        started_at = time.monotonic()
        if started_at < self._cooldown_until:
            return None, "openai_cooldown_active"
        deadline = started_at + total_budget_seconds
        for _ in range(attempts):
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    last_error = "openai_timeout"
                    break
                self._last_request_attempts += 1
                data = self._post_json(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    payload,
                    timeout_seconds=max(0.25, remaining),
                )
                if isinstance(data, dict):
                    self._cooldown_until = 0.0
                    return data, None
                return None, "invalid_api_payload"
            except (TimeoutError, socket.timeout):
                last_error = "openai_timeout"
                self._arm_cooldown(total_budget_seconds)
            except HTTPError as exc:
                last_error = self._http_error_message(exc)
                if 500 <= int(exc.code) < 600:
                    self._arm_cooldown(total_budget_seconds)
                if exc.code in {401, 403, 404}:
                    break
            except URLError:
                last_error = "openai_network_error"
                self._arm_cooldown(total_budget_seconds)
            except Exception:
                last_error = "openai_request_failed"
                self._arm_cooldown(total_budget_seconds)
        return None, last_error or "openai_request_failed"

    def _post_json(self, url: str, payload: dict[str, Any], timeout_seconds: float | None = None) -> dict[str, Any]:
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError("missing_api_key")
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            url=url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        effective_timeout = max(0.25, float(timeout_seconds if timeout_seconds is not None else self.timeout_seconds))
        with urlopen(request, timeout=effective_timeout) as response:  # nosec B310
            body = response.read().decode("utf-8")
        parsed = _safe_json_loads(body)
        if parsed is None:
            raise RuntimeError("invalid_json_body")
        return parsed

    def _get_json(self, url: str, timeout_seconds: float | None = None) -> dict[str, Any]:
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError("missing_api_key")
        request = Request(
            url=url,
            method="GET",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
        )
        effective_timeout = max(0.25, float(timeout_seconds if timeout_seconds is not None else self.timeout_seconds))
        with urlopen(request, timeout=effective_timeout) as response:  # nosec B310
            body = response.read().decode("utf-8")
        parsed = _safe_json_loads(body)
        if parsed is None:
            raise RuntimeError("invalid_json_body")
        return parsed

    @staticmethod
    def _message_content(payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return ""
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if not isinstance(message, dict):
            return ""
        return str(message.get("content", ""))

    @staticmethod
    def _response_status(payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("status") or "").strip().lower()

    @staticmethod
    def _response_error(payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return "openai_request_failed"
        error = payload.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or "").strip().lower()
            message = str(error.get("message") or "").strip().lower()
            if code in {"rate_limit_exceeded", "rate_limited"}:
                return "openai_rate_limited"
            if code in {"invalid_api_key", "authentication_error"}:
                return "openai_auth_failed"
            if "timeout" in message:
                return "openai_timeout"
        status = str(payload.get("status") or "").strip().lower()
        if status == "incomplete":
            incomplete = payload.get("incomplete_details")
            if isinstance(incomplete, dict):
                reason = str(incomplete.get("reason") or "").strip().lower()
                if "max_output_tokens" in reason:
                    return "openai_incomplete_max_output"
        return "openai_request_failed"

    @classmethod
    def _response_text(cls, payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return ""
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        output = payload.get("output")
        if not isinstance(output, list):
            return ""
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type") or "").strip().lower()
                text_value = part.get("text")
                if isinstance(text_value, str) and part_type in {"output_text", "text"}:
                    chunks.append(text_value)
        return "\n".join(chunk for chunk in chunks if chunk.strip()).strip()

    @staticmethod
    def _http_error_message(exc: HTTPError) -> str:
        if exc.code == 429:
            return "openai_rate_limited"
        if exc.code in {401, 403}:
            return "openai_auth_failed"
        if exc.code == 404:
            return "openai_model_not_found"
        if 500 <= int(exc.code) < 600:
            return "openai_server_error"
        return f"openai_http_{exc.code}"

    def _api_key(self) -> str:
        return os.getenv(self.api_key_env, "").strip()

    def _arm_cooldown(self, total_budget_seconds: float) -> None:
        self._cooldown_until = max(self._cooldown_until, time.monotonic() + max(2.0, float(total_budget_seconds)))

    def _set_mode(self, mode: str, error: str | None) -> None:
        self._last_error = error
        if mode != self._last_mode:
            self._last_mode = mode
            logger = self.logger
            if logger is not None:
                if mode == "remote" and hasattr(logger, "info"):
                    logger.info("openai_mode_remote")
                    return
                if mode != "remote" and hasattr(logger, "warning"):
                    logger.warning("openai_mode_fallback", extra={"extra_fields": {"reason": error or "unknown"}})
