from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.apex_telegram import (
    ApexTelegramError,
    client_from_env,
    owner_chat_id_from_env,
    telegram_config_from_mapping,
)
from src.config_loader import load_settings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the live APEX Telegram bot using env-only secrets.")
    parser.add_argument("--get-me", action="store_true", help="Verify TELEGRAM_BOT_TOKEN with Telegram getMe")
    parser.add_argument("--discover-chat", action="store_true", help="Print chat IDs from recent bot updates")
    parser.add_argument("--send-test", action="store_true", help="Send a live test message to TELEGRAM_CHAT_ID or a discovered chat")
    parser.add_argument("--message", default="", help="Optional test message body")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = load_settings(ROOT)
    telegram_config = telegram_config_from_mapping(settings.raw.get("telegram", {}))
    result: dict[str, object] = {
        "enabled": telegram_config.enabled,
        "bot_username": telegram_config.bot_username,
        "token_env": telegram_config.token_env,
        "chat_id_env": telegram_config.owner_chat_id_env,
        "webhook_secret_env": telegram_config.webhook_secret_env,
    }

    try:
        client = client_from_env(telegram_config)
        if args.get_me or not any((args.discover_chat, args.send_test)):
            me = client.get_me()
            result["bot"] = {
                "id": me.get("id"),
                "username": me.get("username"),
                "is_bot": me.get("is_bot"),
            }
        if args.discover_chat:
            chat_ids = client.discover_chat_ids()
            result["chat_ids"] = chat_ids
            result["chat_count"] = len(chat_ids)
        if args.send_test:
            chat_id = owner_chat_id_from_env(telegram_config, client)
            if not chat_id:
                raise ApexTelegramError(
                    f"Missing {telegram_config.owner_chat_id_env}; send /start to the bot, then rerun with --discover-chat"
                )
            text = args.message.strip() or "APEX Telegram live test OK. Dashboard bridge controls are wired."
            sent = client.send_message(chat_id, text, parse_mode=telegram_config.parse_mode)
            result["sent"] = bool(sent.get("message_id"))
            result["chat_id"] = chat_id
    except ApexTelegramError as exc:
        result["ok"] = False
        result["error"] = str(exc)
        print(json.dumps(result, sort_keys=True))
        return 1

    result["ok"] = True
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
