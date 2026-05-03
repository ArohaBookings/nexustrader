from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.apex_telegram import ApexTelegramError, client_from_env, extract_telegram_message, telegram_config_from_mapping
from src.config_loader import load_settings


BOT_COMMANDS = [
    {"command": "start", "description": "Link and show help"},
    {"command": "status", "description": "Live bot status"},
    {"command": "funded", "description": "Funded account guardrails"},
    {"command": "risk", "description": "Risk and circuit breakers"},
    {"command": "trades", "description": "Recent trades and orders"},
    {"command": "aggression", "description": "Live aggression tier and unlock flow"},
    {"command": "apex", "description": "Bot diagnostics and scaling state"},
    {"command": "pause", "description": "Pause trading"},
    {"command": "resume", "description": "Resume trading with confirmation"},
    {"command": "kill", "description": "Kill switch with confirmation"},
    {"command": "refresh", "description": "Refresh bridge state"},
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Telegram polling sidecar for APEX Mac/EA bridge mode.")
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    parser.add_argument("--claim-owner", action="store_true", help="Set TELEGRAM_CHAT_ID from the first incoming chat when missing")
    parser.add_argument("--timeout", type=int, default=25, help="Telegram long-poll timeout seconds")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep between polling loops")
    parser.add_argument("--bridge-url", default="", help="Override local bridge base URL, for example http://127.0.0.1:8000")
    return parser.parse_args()


def _offset_path() -> Path:
    path = ROOT / "data" / "telegram_poll_offset.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _read_offset() -> int | None:
    path = _offset_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = int(payload.get("offset"))
        return value if value > 0 else None
    except Exception:
        return None


def _write_offset(offset: int) -> None:
    _offset_path().write_text(json.dumps({"offset": int(offset)}, sort_keys=True) + "\n", encoding="utf-8")


def _secrets_path() -> Path:
    return ROOT / "config" / "secrets.env"


def _set_secret_key(key: str, value: str) -> None:
    path = _secrets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                raw_key, raw_value = line.split("=", 1)
                if raw_key.strip():
                    entries[raw_key.strip()] = raw_value.strip()
    entries[key] = value
    order = [
        "LIVE_TRADING",
        "APEX_MT5_RUNTIME_MODE",
        "MT5_LOGIN",
        "MT5_PASSWORD",
        "MT5_SERVER",
        "MT5_TERMINAL_PATH",
        "MT5_PATH",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "TELEGRAM_WEBHOOK_SECRET",
        "OPENAI_API_KEY",
        "NEWS_API_KEY",
        "FINNHUB_API_KEY",
        "TRADINGECONOMICS_API_KEY",
    ]
    for existing_key in entries:
        if existing_key not in order:
            order.append(existing_key)
    lines = ["# Local runtime secrets. Do not commit real values."]
    for item in order:
        if item in entries and entries[item] != "":
            lines.append(f"{item}={entries[item]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _bridge_url(settings: Any, override: str) -> str:
    if override:
        return override.rstrip("/")
    bridge = settings.section("bridge") if isinstance(settings.raw.get("bridge"), dict) else {}
    host = str(os.getenv("BRIDGE_HOST") or bridge.get("host") or "127.0.0.1").strip()
    port = int(os.getenv("BRIDGE_PORT") or bridge.get("port") or 8000)
    return f"http://{host}:{port}"


def _post_to_bridge(base_url: str, secret: str, update: dict[str, Any]) -> tuple[bool, str]:
    request = Request(
        f"{base_url.rstrip('/')}/telegram/webhook",
        data=json.dumps(update).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Telegram-Bot-Api-Secret-Token": secret,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=4.0) as response:  # nosec B310
            return int(getattr(response, "status", 500)) < 500, "forwarded"
    except URLError as exc:
        return False, f"bridge_unreachable:{exc.reason}"
    except Exception as exc:
        return False, f"bridge_forward_failed:{exc}"


def main() -> int:
    args = _parse_args()
    settings = load_settings(ROOT)
    config = telegram_config_from_mapping(settings.raw.get("telegram", {}))
    client = client_from_env(config)
    client.delete_webhook(drop_pending_updates=False)
    client.set_my_commands(BOT_COMMANDS)
    secret = os.getenv(config.webhook_secret_env, "").strip()
    if not secret:
        raise SystemExit(f"Missing {config.webhook_secret_env}")
    base_url = _bridge_url(settings, args.bridge_url)
    offset = _read_offset()
    print(json.dumps({"ok": True, "mode": "polling", "bridge_url": base_url, "offset": offset}, sort_keys=True))

    while True:
        try:
            updates = client.get_updates(timeout=max(0, int(args.timeout)), limit=50, offset=offset)
        except ApexTelegramError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
            if args.once:
                return 1
            time.sleep(max(1.0, float(args.sleep)))
            continue

        for update in updates:
            update_id = int(update.get("update_id", 0))
            offset = max(offset or 0, update_id + 1)
            message = extract_telegram_message(update)
            if message is not None and args.claim_owner and not os.getenv(config.owner_chat_id_env, "").strip():
                _set_secret_key(config.owner_chat_id_env, message.chat_id)
                os.environ[config.owner_chat_id_env] = message.chat_id
                client.send_message(message.chat_id, "APEX owner chat linked. Local polling is active.", parse_mode=config.parse_mode)
            ok, reason = _post_to_bridge(base_url, secret, update)
            if message is not None and not ok:
                client.send_message(
                    message.chat_id,
                    f"APEX bridge is not reachable yet: <code>{reason}</code>",
                    parse_mode=config.parse_mode,
                )
            print(json.dumps({"update_id": update_id, "forwarded": ok, "reason": reason}, sort_keys=True))
            _write_offset(offset)

        if args.once:
            return 0
        time.sleep(max(0.2, float(args.sleep)))


if __name__ == "__main__":
    raise SystemExit(main())
