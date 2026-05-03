from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_loader import load_env_files
from src.hyperliquid_lab import load_lab_config
from src.hyperliquid_lab.telegram import (
    TelegramBotError,
    TelegramCommandRouter,
    TelegramNotifier,
    build_lab_overview,
    client_from_env,
    notifier_from_env,
    telegram_config_from_lab_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram utilities for the Hyperliquid lab.")
    parser.add_argument("--get-me", action="store_true", help="Verify token and print bot identity")
    parser.add_argument("--discover-chat", action="store_true", help="Print chat IDs from recent updates")
    parser.add_argument("--send-test", action="store_true", help="Send a test overview message")
    parser.add_argument("--poll-once", action="store_true", help="Handle current command updates once")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_lab_config()
    load_env_files(config.project_root)
    telegram_config = telegram_config_from_lab_config(config)
    try:
        client = client_from_env(telegram_config)
        if args.get_me:
            me = client.get_me()
            print({"id": me.get("id"), "username": me.get("username"), "is_bot": me.get("is_bot")})
        if args.discover_chat:
            chat_ids = client.discover_chat_ids()
            print({"chat_ids": chat_ids, "count": len(chat_ids)})
        if args.send_test:
            notifier = notifier_from_env(telegram_config)
            notifier.send_test(build_lab_overview(config))
            print({"sent": True})
        if args.poll_once:
            def _factory(chat_id):
                return TelegramNotifier(client=client, chat_id=chat_id, parse_mode=telegram_config.parse_mode)

            router = TelegramCommandRouter(_factory, lambda: build_lab_overview(config))
            handled = 0
            for update in client.get_updates(timeout=0, limit=100):
                handled += 1 if router.handle_update(update) else 0
            print({"handled": handled})
    except TelegramBotError as exc:
        print({"ok": False, "error": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
