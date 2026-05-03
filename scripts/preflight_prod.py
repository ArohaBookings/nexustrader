#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from src.config_loader import load_settings
from src.logger import LoggerFactory
from src.mt5_client import MT5Client, MT5Credentials


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        target = path / ".apex_preflight_write"
        target.write_text("ok", encoding="utf-8")
        target.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def main() -> int:
    settings = load_settings()
    logging_config = settings.section("logging") if isinstance(settings.raw.get("logging"), dict) else {}
    logger = LoggerFactory(
        log_file=settings.runtime_paths.logs_dir / "apex.log",
        rotate_max_bytes=int(logging_config.get("rotate_max_bytes", 10 * 1024 * 1024)),
        rotate_backup_count=int(logging_config.get("rotate_backup_count", 7)),
    ).build()
    system_config = settings.section("system")
    bridge_config = settings.section("bridge") if isinstance(settings.raw.get("bridge"), dict) else {}
    ai_config = settings.section("ai")
    news_config = settings.section("news")

    mt5_client = MT5Client(
        credentials=MT5Credentials.from_env(default_terminal_path=str(system_config.get("mt5_terminal_path", "") or "") or None),
        journal_db=settings.path("data.trade_db"),
        logger=logger,
        disable_mt5=False,
        symbol_mapping=system_config.get("symbol_mapping", {}) if isinstance(system_config.get("symbol_mapping"), dict) else {},
    )
    verification = mt5_client.verify_connection(settings.symbols())
    if mt5_client.connected:
        mt5_client.shutdown()

    bridge_url = f"http://{bridge_config.get('host', '127.0.0.1')}:{int(bridge_config.get('port', 8000))}/health"
    bridge_reachable = False
    try:
        with urlopen(bridge_url, timeout=2.0) as response:
            bridge_reachable = int(getattr(response, "status", 500)) == 200
    except URLError:
        bridge_reachable = False

    payload = {
        "ok": bool(verification.get("ok", False)),
        "mt5_reachable": bool(verification.get("ok", False)),
        "bridge_reachable": bridge_reachable,
        "account_connected": bool(verification.get("account")),
        "runtime_paths": settings.runtime_paths.snapshot(),
        "logs_writable": _writable(settings.runtime_paths.logs_dir),
        "cache_writable": _writable(settings.runtime_paths.cache_dir),
        "config_loaded": True,
        "ai_key_present": bool(os.getenv(str(ai_config.get("openai_api_env", "OPENAI_API_KEY")), "").strip()),
        "public_news_proxy_available": bool(
            str(news_config.get("provider", "")).strip().lower() in {"stub", "safe", "disabled"}
            or os.getenv(str(news_config.get("api_key_env", "")), "").strip()
            or str(news_config.get("api_key", "")).strip()
            or os.getenv(str(news_config.get("fallback_api_key_env", "")), "").strip()
            or str(news_config.get("fallback_api_key", "")).strip()
        ),
        "startup_sequence_status": "ready" if verification.get("ok", False) else "awaiting_mt5",
        "restart_safe_reconciliation_status": "journal_scoped_to_live_snapshot",
        "symbols": verification.get("symbols", {}),
        "account": verification.get("account"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
