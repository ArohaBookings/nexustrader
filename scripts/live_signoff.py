from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.apex_telegram import ApexTelegramError, client_from_env, telegram_config_from_mapping
from src.config_loader import load_settings
from src.env_loader import load_env_files
from src.live_readiness import (
    build_live_readiness_report,
    collect_bridge_health,
    collect_gh_auth_probe,
    collect_git_probe,
    collect_vercel_probe,
)
from src.logger import LoggerFactory
from src.mt5_client import MT5Client, MT5Credentials


def _ensure_user_tool_paths() -> None:
    candidates = [
        Path.home() / ".npm-global" / "bin",
        Path.home() / ".local" / "bin",
        Path("/opt/homebrew/bin"),
    ]
    existing = [str(path) for path in candidates if path.exists()]
    if not existing:
        return
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    for item in reversed(existing):
        if item not in parts:
            parts.insert(0, item)
    os.environ["PATH"] = os.pathsep.join(parts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard sign-off gate for APEX live trading, Telegram, GitHub, and Vercel readiness.")
    parser.add_argument("--skip-mt5", action="store_true", help="Do not run MT5 initialize check; report remains blocked for MT5")
    parser.add_argument("--skip-bridge", action="store_true", help="Do not call bridge /health; report remains blocked for bridge")
    parser.add_argument("--no-deploy", action="store_true", help="Do not treat GitHub/Vercel deployment readiness as a hard blocker")
    parser.add_argument("--send-telegram-test", action="store_true", help="Send a short Telegram test message if token/chat are configured")
    return parser.parse_args()


def _mt5_verify(settings: Any) -> dict[str, Any]:
    logging_config = settings.section("logging") if isinstance(settings.raw.get("logging"), dict) else {}
    logger = LoggerFactory(
        log_file=settings.runtime_paths.logs_dir / "apex.log",
        rotate_max_bytes=int(logging_config.get("rotate_max_bytes", 10 * 1024 * 1024)),
        rotate_backup_count=int(logging_config.get("rotate_backup_count", 7)),
        retention_days=int(logging_config.get("retention_days", 365)),
    ).build()
    system_config = settings.section("system")
    client = MT5Client(
        credentials=MT5Credentials.from_env(default_terminal_path=str(system_config.get("mt5_terminal_path", "") or "") or None),
        journal_db=settings.path("data.trade_db"),
        logger=logger,
        disable_mt5=False,
        symbol_mapping=system_config.get("symbol_mapping", {}) if isinstance(system_config.get("symbol_mapping"), dict) else {},
    )
    try:
        return client.verify_connection(settings.symbols())
    finally:
        if client.connected:
            client.shutdown()


def _telegram_identity(settings: Any, *, send_test: bool) -> dict[str, Any]:
    telegram_config = telegram_config_from_mapping(settings.raw.get("telegram", {}))
    try:
        client = client_from_env(telegram_config)
        me = client.get_me()
        identity = {"ok": bool(me.get("is_bot")), "id": me.get("id"), "username": me.get("username"), "is_bot": me.get("is_bot")}
        if send_test:
            chat_id = os.getenv(telegram_config.owner_chat_id_env, "").strip()
            if not chat_id:
                identity["test_sent"] = False
                identity["test_error"] = f"missing {telegram_config.owner_chat_id_env}"
            else:
                sent = client.send_message(chat_id, "APEX live sign-off Telegram test OK.", parse_mode=telegram_config.parse_mode)
                identity["test_sent"] = bool(sent.get("message_id"))
        return identity
    except ApexTelegramError as exc:
        return {"ok": False, "error": str(exc)}


def _ea_bridge_mode() -> bool:
    mode = str(os.getenv("APEX_MT5_RUNTIME_MODE") or os.getenv("MT5_RUNTIME_MODE") or "").strip().upper()
    return mode in {"EA_BRIDGE", "EA", "BRIDGE", "WEBREQUEST"}


def main() -> int:
    _ensure_user_tool_paths()
    args = _parse_args()
    load_env_files(ROOT)
    settings = load_settings(ROOT)
    bridge_config = settings.section("bridge") if isinstance(settings.raw.get("bridge"), dict) else {}
    dashboard_config = settings.raw.get("dashboard", {}) if isinstance(settings.raw.get("dashboard"), dict) else {}
    host = str(bridge_config.get("host", "127.0.0.1") or "127.0.0.1")
    port = int(bridge_config.get("port", 8000) or 8000)
    if bool(dashboard_config.get("public_enabled", False)):
        host = "127.0.0.1"
        port = int(dashboard_config.get("bind_port") or dashboard_config.get("port") or port)

    if args.skip_mt5:
        mt5 = {"ok": False, "reasons": ["mt5_check_skipped"]}
    elif _ea_bridge_mode():
        mt5 = {
            "ok": True,
            "connected": False,
            "runtime_mode": "EA_BRIDGE",
            "reasons": ["mt5_python_skipped_ea_bridge_mode"],
        }
    else:
        mt5 = _mt5_verify(settings)
    bridge = {"ok": False, "error": "bridge_check_skipped"} if args.skip_bridge else collect_bridge_health(host, port)
    report = build_live_readiness_report(
        project_root=ROOT,
        settings_raw=settings.raw,
        env=os.environ,
        git_probe=collect_git_probe(ROOT),
        gh_auth_probe=collect_gh_auth_probe(ROOT) if shutil.which("gh") else {"ok": False, "error": "gh missing"},
        vercel_probe=collect_vercel_probe(ROOT) if shutil.which("vercel") else {"ok": False, "error": "vercel missing"},
        mt5_verification=mt5,
        bridge_health=bridge,
        telegram_identity=_telegram_identity(settings, send_test=bool(args.send_telegram_test)),
        require_deploy=not bool(args.no_deploy),
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if bool(report.get("ready")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
