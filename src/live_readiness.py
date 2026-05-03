from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import json
import os
import shutil
import subprocess
from urllib.error import URLError
from urllib.request import urlopen


CommandExists = Callable[[str], bool]
CommandRunner = Callable[[Sequence[str], Path], tuple[int, str]]


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    hard: bool
    summary: str
    details: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "hard": self.hard,
            "summary": self.summary,
            "details": dict(self.details),
        }


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_present(env: Mapping[str, str], key: str) -> bool:
    return bool(str(env.get(key, "")).strip())


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else default


def _check(name: str, ok: bool, summary: str, *, hard: bool = True, details: Mapping[str, Any] | None = None) -> ReadinessCheck:
    return ReadinessCheck(name=name, status="PASS" if ok else "FAIL", hard=hard, summary=summary, details=dict(details or {}))


def _warn(name: str, summary: str, *, details: Mapping[str, Any] | None = None) -> ReadinessCheck:
    return ReadinessCheck(name=name, status="WARN", hard=False, summary=summary, details=dict(details or {}))


def _default_command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def _default_command_runner(args: Sequence[str], cwd: Path) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except FileNotFoundError as exc:
        return 127, str(exc)
    except subprocess.TimeoutExpired:
        return 124, "command timed out"
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part and part.strip())
    return int(completed.returncode), output


def collect_git_probe(root: Path, command_runner: CommandRunner = _default_command_runner) -> dict[str, Any]:
    repo_code, repo_output = command_runner(("git", "rev-parse", "--is-inside-work-tree"), root)
    if repo_code != 0 or repo_output.strip().lower() != "true":
        return {"is_repo": False, "error": repo_output}
    remote_code, remote_output = command_runner(("git", "remote", "get-url", "origin"), root)
    branch_code, branch_output = command_runner(("git", "branch", "--show-current"), root)
    head_code, head_output = command_runner(("git", "rev-parse", "--verify", "HEAD"), root)
    status_code, status_output = command_runner(("git", "status", "--porcelain"), root)
    upstream_code, upstream_output = command_runner(("git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"), root)
    dirty_lines = [line for line in status_output.splitlines() if line.strip()] if status_code == 0 else []
    return {
        "is_repo": True,
        "remote": remote_output if remote_code == 0 else "",
        "branch": branch_output if branch_code == 0 else "",
        "upstream": upstream_output if upstream_code == 0 else "",
        "has_head": head_code == 0,
        "head": head_output if head_code == 0 else "",
        "dirty_count": len(dirty_lines),
        "dirty_preview": dirty_lines[:12],
    }


def collect_bridge_health(host: str, port: int, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
    url = f"http://{host}:{int(port)}/health"
    try:
        with urlopen(url, timeout=max(0.5, float(timeout_seconds))) as response:  # nosec B310
            raw = response.read().decode("utf-8")
            payload = json.loads(raw)
            return payload if isinstance(payload, dict) else {"ok": False, "error": "unexpected payload"}
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc), "url": url}


def build_live_readiness_report(
    *,
    project_root: Path,
    settings_raw: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
    command_exists: CommandExists | None = None,
    git_probe: Mapping[str, Any] | None = None,
    gh_auth_probe: Mapping[str, Any] | None = None,
    vercel_probe: Mapping[str, Any] | None = None,
    mt5_verification: Mapping[str, Any] | None = None,
    bridge_health: Mapping[str, Any] | None = None,
    telegram_identity: Mapping[str, Any] | None = None,
    require_deploy: bool = True,
) -> dict[str, Any]:
    env_map = env if env is not None else os.environ
    exists = command_exists or _default_command_exists
    root = project_root.resolve()
    checks: list[ReadinessCheck] = []

    system = settings_raw.get("system", {}) if isinstance(settings_raw.get("system"), Mapping) else {}
    dashboard = settings_raw.get("dashboard", {}) if isinstance(settings_raw.get("dashboard"), Mapping) else {}
    telegram = settings_raw.get("telegram", {}) if isinstance(settings_raw.get("telegram"), Mapping) else {}
    bridge = settings_raw.get("bridge", {}) if isinstance(settings_raw.get("bridge"), Mapping) else {}
    ai = settings_raw.get("ai", {}) if isinstance(settings_raw.get("ai"), Mapping) else {}
    news = settings_raw.get("news", {}) if isinstance(settings_raw.get("news"), Mapping) else {}

    token_env = str(telegram.get("token_env", "TELEGRAM_BOT_TOKEN"))
    chat_env = str(telegram.get("owner_chat_id_env", telegram.get("chat_id_env", "TELEGRAM_CHAT_ID")))
    webhook_secret_env = str(telegram.get("webhook_secret_env", "TELEGRAM_WEBHOOK_SECRET"))
    openai_env = str(ai.get("openai_api_env", "OPENAI_API_KEY"))

    for command in ("git", "gh", "vercel"):
        checks.append(
            _check(
                f"cli_{command}",
                exists(command),
                f"{command} CLI is available" if exists(command) else f"{command} CLI is missing",
                hard=require_deploy,
                details={"command": command},
            )
        )

    expected_remote = "github.com/ArohaBookings/nexustrader"
    git_payload = dict(git_probe or {})
    remote = str(git_payload.get("remote") or "")
    checks.append(
        _check(
            "git_repository",
            bool(git_payload.get("is_repo")) and expected_remote.lower() in remote.lower().replace(":", "/"),
            "Root is connected to ArohaBookings/nexustrader"
            if bool(git_payload.get("is_repo")) and expected_remote.lower() in remote.lower().replace(":", "/")
            else "Root is not a git checkout connected to ArohaBookings/nexustrader",
            hard=require_deploy,
            details={"remote": remote, "branch": str(git_payload.get("branch") or ""), "is_repo": bool(git_payload.get("is_repo"))},
        )
    )
    if bool(git_payload.get("is_repo")):
        has_head = bool(git_payload.get("has_head", True))
        dirty_count = int(_number(git_payload.get("dirty_count"), 0.0))
        checks.append(
            _check(
                "git_worktree_clean",
                has_head and dirty_count == 0,
                "Git worktree is committed and clean" if has_head and dirty_count == 0 else "Git worktree has uncommitted or untracked changes",
                hard=require_deploy,
                details={
                    "has_head": has_head,
                    "dirty_count": dirty_count,
                    "dirty_preview": list(git_payload.get("dirty_preview") or [])[:12],
                },
            )
        )
        upstream = str(git_payload.get("upstream") or "").strip()
        checks.append(
            _check(
                "git_branch_published",
                bool(upstream),
                "Git branch has an upstream remote branch" if upstream else "Git branch is not published to an upstream remote branch",
                hard=require_deploy,
                details={"branch": str(git_payload.get("branch") or ""), "upstream": upstream},
            )
        )

    gh_payload = dict(gh_auth_probe or {})
    github_publish_ready = bool(gh_payload.get("ok")) or bool(git_payload.get("upstream"))
    checks.append(
        _check(
            "github_auth",
            github_publish_ready,
            "GitHub publish path is available" if github_publish_ready else "GitHub CLI is not authenticated and branch is not published",
            hard=require_deploy,
            details={**{k: v for k, v in gh_payload.items() if k != "token"}, "upstream": str(git_payload.get("upstream") or "")},
        )
    )

    vercel_payload = dict(vercel_probe or {})
    checks.append(
        _check(
            "vercel_auth_project",
            bool(vercel_payload.get("ok")),
            "Vercel CLI/project link is ready" if bool(vercel_payload.get("ok")) else "Vercel CLI/project link is not ready",
            hard=require_deploy,
            details=vercel_payload,
        )
    )
    checks.append(
        _warn(
            "vercel_runtime_architecture",
            "Vercel can host dashboard/webhook surfaces, but the MT5 execution worker must run on the MT5 VPS/VM.",
            details={"reason": "MT5 requires a terminal process; serverless deploys are not a substitute for the live broker bridge."},
        )
    )

    required_env = (token_env, chat_env, webhook_secret_env, openai_env)
    missing_env = [key for key in required_env if not _env_present(env_map, key)]
    checks.append(
        _check(
            "required_env",
            not missing_env,
            "Required secrets are present" if not missing_env else "Required secrets are missing",
            hard=True,
            details={"missing": missing_env, "checked": list(required_env)},
        )
    )

    news_env_keys = [
        str(news.get("api_key_env", "") or ""),
        str(news.get("fallback_api_key_env", "") or ""),
    ]
    news_key_present = any(key and _env_present(env_map, key) for key in news_env_keys) or bool(str(news.get("api_key", "") or "").strip())
    checks.append(
        _check(
            "news_provider",
            news_key_present or str(news.get("provider", "")).lower() in {"stub", "safe", "disabled"},
            "News provider key/fallback is available" if news_key_present else "News provider key/fallback is missing",
            hard=False,
            details={"provider": str(news.get("provider", "")), "env_keys_checked": [key for key in news_env_keys if key]},
        )
    )

    mt5_payload = dict(mt5_verification or {})
    checks.append(
        _check(
            "mt5_connection",
            bool(mt5_payload.get("ok")),
            "MT5 terminal/account is connected" if bool(mt5_payload.get("ok")) else "MT5 terminal/account is not connected",
            hard=True,
            details={
                "reasons": list(mt5_payload.get("reasons") or []),
                "account": mt5_payload.get("account") or mt5_payload.get("account_summary"),
                "version": mt5_payload.get("version"),
            },
        )
    )

    bridge_payload = dict(bridge_health or {})
    broker = bridge_payload.get("broker_connectivity") if isinstance(bridge_payload.get("broker_connectivity"), Mapping) else {}
    checks.append(
        _check(
            "bridge_health",
            bool(bridge_payload.get("ok")) and str(bridge_payload.get("bridge_status", "")).upper() == "UP",
            "Bridge health endpoint is up" if bool(bridge_payload.get("ok")) else "Bridge health endpoint is unavailable",
            hard=True,
            details={
                "bridge_status": bridge_payload.get("bridge_status"),
                "terminal_connected": bool(broker.get("terminal_connected")) if isinstance(broker, Mapping) else False,
                "account": broker.get("account") if isinstance(broker, Mapping) else "",
            },
        )
    )
    checks.append(
        _check(
            "bridge_mt5_feed",
            bool(broker.get("terminal_connected")) and bool(broker.get("terminal_trade_allowed")) if isinstance(broker, Mapping) else False,
            "Bridge is receiving live MT5 terminal/trade-allowed state"
            if isinstance(broker, Mapping) and bool(broker.get("terminal_connected")) and bool(broker.get("terminal_trade_allowed"))
            else "Bridge is not receiving live MT5 trade-allowed state",
            hard=True,
            details=dict(broker) if isinstance(broker, Mapping) else {},
        )
    )

    identity = dict(telegram_identity or {})
    checks.append(
        _check(
            "telegram_bot_identity",
            bool(identity.get("ok")),
            "Telegram bot token verified with getMe" if bool(identity.get("ok")) else "Telegram bot token was not verified",
            hard=True,
            details={k: v for k, v in identity.items() if k != "token"},
        )
    )

    live_trading_enabled = bool(system.get("live_trading", False)) and bool(system.get("trading_enabled", False))
    checks.append(
        _check(
            "live_trading_config",
            live_trading_enabled,
            "Live trading flags are enabled" if live_trading_enabled else "Live trading flags are disabled",
            hard=False,
            details={
                "system.live_trading": bool(system.get("live_trading", False)),
                "system.trading_enabled": bool(system.get("trading_enabled", False)),
            },
        )
    )

    if bool(dashboard.get("public_enabled", False)):
        checks.append(
            _check(
                "public_dashboard_guard",
                bool(str(dashboard.get("password", "")).strip()),
                "Public dashboard has password protection configured",
                hard=True,
                details={"read_only": bool(dashboard.get("read_only", True)), "allowed_ips_count": len(dashboard.get("allowed_ips", []) or [])},
            )
        )

    hard_failures = [check for check in checks if check.hard and check.status == "FAIL"]
    warnings = [check for check in checks if check.status == "WARN" or (not check.hard and check.status == "FAIL")]
    ready = not hard_failures
    next_actions = _next_actions_from_failures(hard_failures, warnings)
    return {
        "generated_at": _utc_iso(),
        "project_root": str(root),
        "overall_status": "READY" if ready else "BLOCKED",
        "ready": ready,
        "hard_blocker_count": len(hard_failures),
        "warning_count": len(warnings),
        "hard_blockers": [check.as_dict() for check in hard_failures],
        "warnings": [check.as_dict() for check in warnings],
        "checks": [check.as_dict() for check in checks],
        "next_actions": next_actions,
    }


def _next_actions_from_failures(hard_failures: Sequence[ReadinessCheck], warnings: Sequence[ReadinessCheck]) -> list[str]:
    names = {check.name for check in hard_failures}
    actions: list[str] = []
    if "mt5_connection" in names or "bridge_mt5_feed" in names:
        actions.append("Run this sign-off on the Windows MT5 VPS/VM with MetaTrader 5 installed, logged in, and WebRequest enabled for the bridge URL.")
    if "required_env" in names:
        actions.append("Populate config/secrets.env with Telegram, webhook, and OpenAI secrets; never commit the file.")
    if "telegram_bot_identity" in names:
        actions.append("Run scripts/apex_telegram_check.py --get-me, then send /start to the bot and run --discover-chat/--send-test.")
    if "git_repository" in names or "github_auth" in names or "cli_gh" in names:
        actions.append("Use a real git checkout of https://github.com/ArohaBookings/nexustrader.git and authenticate gh before pushing.")
    if "git_worktree_clean" in names:
        actions.append("Commit or intentionally discard local source changes before deployment sign-off.")
    if "git_branch_published" in names:
        actions.append("Push the active branch to GitHub and set upstream before Vercel/GitHub deployment sign-off.")
    if "vercel_auth_project" in names or "cli_vercel" in names:
        actions.append("Install/authenticate Vercel CLI and link the dashboard/webhook project before production deploy.")
    if any(check.name == "vercel_runtime_architecture" for check in warnings):
        actions.append("Keep MT5 execution on the VPS/VM; use Vercel only for dashboard/webhook surfaces or a secure proxy.")
    return actions


def collect_gh_auth_probe(root: Path, command_runner: CommandRunner = _default_command_runner) -> dict[str, Any]:
    code, output = command_runner(("gh", "auth", "status", "--active", "--hostname", "github.com"), root)
    return {"ok": code == 0, "output": output[:1000]}


def collect_vercel_probe(root: Path, command_runner: CommandRunner = _default_command_runner) -> dict[str, Any]:
    version_code, version_output = command_runner(("vercel", "--version"), root)
    project_file = root / ".vercel" / "project.json"
    token_present = _env_present(os.environ, "VERCEL_TOKEN")
    project_env_present = _env_present(os.environ, "VERCEL_ORG_ID") and _env_present(os.environ, "VERCEL_PROJECT_ID")
    linked = project_file.exists() or project_env_present
    return {
        "ok": version_code == 0 and linked and (token_present or project_file.exists()),
        "version": version_output if version_code == 0 else "",
        "project_linked": linked,
        "token_present": token_present,
        "project_env_present": project_env_present,
    }
