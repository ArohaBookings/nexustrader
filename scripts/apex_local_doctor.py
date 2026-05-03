from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.apex_telegram import ApexTelegramError, client_from_env, telegram_config_from_mapping
from src.config_loader import load_settings
from src.env_loader import load_env_files
from src.live_readiness import collect_bridge_health


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Mac APEX doctor for bridge, MT5 EA, and Telegram readiness.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    parser.add_argument("--send-telegram-test", action="store_true", help="Send a Telegram test message when TELEGRAM_CHAT_ID is configured")
    return parser.parse_args()


def _run(args: list[str], *, timeout: float = 4.0) -> tuple[int, str]:
    try:
        completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        return 127, str(exc)
    except subprocess.TimeoutExpired:
        return 124, "command timed out"
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part and part.strip())
    return int(completed.returncode), output


def _screen_sessions() -> dict[str, Any]:
    code, output = _run(["screen", "-ls"])
    sessions = {
        "apex_bridge": ".apex_bridge" in output,
        "apex_telegram": ".apex_telegram" in output,
    }
    return {"ok": code in {0, 1}, "sessions": sessions, "raw": output}


def _process_status() -> dict[str, Any]:
    _, output = _run(["/bin/sh", "-lc", "ps aux | grep -E 'src.main --bridge-only|start_bridge_prod.py|apex_telegram_poll.py --claim-owner' | grep -v grep || true"])
    return {
        "bridge_process": "src.main --bridge-only" in output or "start_bridge_prod.py" in output,
        "telegram_poller_process": "apex_telegram_poll.py --claim-owner" in output,
        "raw": output,
    }


def _telegram_status(settings: Any, *, send_test: bool) -> dict[str, Any]:
    config = telegram_config_from_mapping(settings.raw.get("telegram", {}))
    chat_id = os.getenv(config.owner_chat_id_env, "").strip()
    payload: dict[str, Any] = {
        "chat_id_present": bool(chat_id),
        "bot_identity_ok": False,
        "test_sent": False,
    }
    try:
        client = client_from_env(config)
        identity = client.get_me()
        payload.update(
            {
                "bot_identity_ok": bool(identity.get("is_bot")),
                "bot_username": identity.get("username"),
                "bot_id": identity.get("id"),
            }
        )
        if send_test and chat_id:
            sent = client.send_message(chat_id, "APEX local doctor Telegram test OK.", parse_mode=config.parse_mode)
            payload["test_sent"] = bool(sent.get("message_id"))
    except ApexTelegramError as exc:
        payload["error"] = str(exc)
    return payload


def _bridge_summary(settings: Any) -> dict[str, Any]:
    bridge_config = settings.section("bridge") if isinstance(settings.raw.get("bridge"), dict) else {}
    dashboard_config = settings.raw.get("dashboard", {}) if isinstance(settings.raw.get("dashboard"), dict) else {}
    host = str(bridge_config.get("host", "127.0.0.1") or "127.0.0.1")
    port = int(bridge_config.get("port", 8000) or 8000)
    if bool(dashboard_config.get("public_enabled", False)):
        host = "127.0.0.1"
        port = int(dashboard_config.get("bind_port") or dashboard_config.get("port") or port)
    health = collect_bridge_health(host, port, timeout_seconds=3.0)
    broker = health.get("broker_connectivity") if isinstance(health.get("broker_connectivity"), dict) else {}
    return {
        "health_ok": bool(health.get("ok")) and str(health.get("bridge_status", "")).upper() == "UP",
        "bridge_status": health.get("bridge_status"),
        "account": broker.get("account"),
        "magic": broker.get("magic"),
        "ea_polling_fresh": bool(broker.get("ea_polling_fresh")),
        "terminal_connected": broker.get("terminal_connected"),
        "terminal_trade_allowed": broker.get("terminal_trade_allowed"),
        "mql_trade_allowed": broker.get("mql_trade_allowed"),
        "explicit_permission_flags": bool(broker.get("explicit_permission_flags")),
        "permission_source": broker.get("permission_source"),
        "last_poll_age_seconds": broker.get("last_poll_age_seconds"),
        "queue_depth": health.get("queue_depth"),
        "raw_health_error": health.get("error"),
    }


def _build_report(*, send_telegram_test: bool) -> dict[str, Any]:
    load_env_files(ROOT)
    settings = load_settings(ROOT)
    screen = _screen_sessions()
    processes = _process_status()
    bridge = _bridge_summary(settings)
    telegram = _telegram_status(settings, send_test=send_telegram_test)
    blockers: list[str] = []
    warnings: list[str] = []

    if not bridge["health_ok"]:
        blockers.append("bridge_health_down")
    if not bridge["ea_polling_fresh"]:
        blockers.append("mt5_ea_not_polling")
    if not telegram["bot_identity_ok"]:
        blockers.append("telegram_bot_identity_failed")
    if not telegram["chat_id_present"]:
        blockers.append("telegram_chat_id_missing")
    if not processes["telegram_poller_process"]:
        blockers.append("telegram_poller_not_running")
    if not processes["bridge_process"]:
        blockers.append("bridge_process_not_running")
    if bridge["ea_polling_fresh"] and not bridge["explicit_permission_flags"]:
        warnings.append("legacy_ea_polling_without_explicit_permission_flags")

    return {
        "ok": not blockers,
        "mode": "local_mac",
        "blockers": blockers,
        "warnings": warnings,
        "screen": screen,
        "processes": processes,
        "bridge": bridge,
        "telegram": telegram,
        "next_actions": _next_actions(blockers, warnings),
    }


def _next_actions(blockers: list[str], warnings: list[str]) -> list[str]:
    actions: list[str] = []
    if "bridge_health_down" in blockers or "bridge_process_not_running" in blockers or "telegram_poller_not_running" in blockers:
        actions.append("./scripts/start_local_mac.sh")
    if "mt5_ea_not_polling" in blockers:
        actions.append("Confirm MT5 has ApexBridgeEA attached and WebRequest allowed for http://127.0.0.1:8000.")
    if "telegram_chat_id_missing" in blockers:
        actions.append("Open https://t.me/Nexus_vantage_trader_bot and send /start.")
    if "legacy_ea_polling_without_explicit_permission_flags" in warnings:
        actions.append("Recompile and reattach mt5_bridge/ApexBridgeEA.mq5 when practical to expose explicit MT5 permission flags.")
    return actions


def _print_human(report: dict[str, Any]) -> None:
    print("APEX LOCAL DOCTOR")
    print(f"Overall: {'READY' if report['ok'] else 'BLOCKED'}")
    print(f"Blockers: {', '.join(report['blockers']) if report['blockers'] else 'none'}")
    print(f"Warnings: {', '.join(report['warnings']) if report['warnings'] else 'none'}")
    bridge = report["bridge"]
    print(
        "Bridge/MT5: "
        f"{bridge.get('bridge_status')} account={bridge.get('account')} "
        f"ea_polling_fresh={bridge.get('ea_polling_fresh')} "
        f"terminal_connected={bridge.get('terminal_connected')} "
        f"last_poll_age={bridge.get('last_poll_age_seconds')}"
    )
    telegram = report["telegram"]
    print(
        "Telegram: "
        f"identity_ok={telegram.get('bot_identity_ok')} "
        f"chat_id_present={telegram.get('chat_id_present')} "
        f"test_sent={telegram.get('test_sent')}"
    )
    for action in report["next_actions"]:
        print(f"Next: {action}")


def main() -> int:
    args = _parse_args()
    report = _build_report(send_telegram_test=bool(args.send_telegram_test))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _print_human(report)
    return 0 if bool(report.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
