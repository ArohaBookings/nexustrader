from __future__ import annotations

from pathlib import Path

from src.live_readiness import build_live_readiness_report


def _settings() -> dict:
    return {
        "system": {"live_trading": True, "trading_enabled": True},
        "telegram": {
            "token_env": "TELEGRAM_BOT_TOKEN",
            "owner_chat_id_env": "TELEGRAM_CHAT_ID",
            "webhook_secret_env": "TELEGRAM_WEBHOOK_SECRET",
        },
        "ai": {"openai_api_env": "OPENAI_API_KEY"},
        "news": {"provider": "finnhub", "api_key_env": "FINNHUB_API_KEY", "fallback_api_key_env": "TRADINGECONOMICS_API_KEY"},
        "dashboard": {"public_enabled": True, "password": "owner-pass", "read_only": True, "allowed_ips": []},
        "bridge": {"host": "127.0.0.1", "port": 8000},
    }


def test_live_readiness_ready_when_all_hard_gates_pass() -> None:
    env = {
        "TELEGRAM_BOT_TOKEN": "set",
        "TELEGRAM_CHAT_ID": "123",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
        "OPENAI_API_KEY": "set",
        "FINNHUB_API_KEY": "set",
        "VERCEL_TOKEN": "set",
    }
    report = build_live_readiness_report(
        project_root=Path("/tmp/apex"),
        settings_raw=_settings(),
        env=env,
        command_exists=lambda command: True,
        git_probe={"is_repo": True, "remote": "git@github.com:ArohaBookings/nexustrader.git", "branch": "main", "upstream": "origin/main"},
        gh_auth_probe={"ok": True},
        vercel_probe={"ok": True, "project_linked": True, "token_present": True},
        mt5_verification={"ok": True, "account": {"login": 1}, "version": "5"},
        bridge_health={
            "ok": True,
            "bridge_status": "UP",
            "broker_connectivity": {"terminal_connected": True, "terminal_trade_allowed": True, "account": "1"},
            "latest_account_snapshot": {
                "account": "1",
                "balance": 100.0,
                "equity": 101.0,
                "free_margin": 95.0,
                "bid": 68000.10,
                "ask": 68000.40,
                "last_price": 68000.25,
            },
        },
        telegram_identity={"ok": True, "username": "Nexus_vantage_trader_bot", "is_bot": True},
    )

    assert report["ready"] is True
    assert report["overall_status"] == "READY"
    assert report["hard_blocker_count"] == 0


def test_live_readiness_accepts_fresh_legacy_ea_polling_with_warning() -> None:
    env = {
        "TELEGRAM_BOT_TOKEN": "set",
        "TELEGRAM_CHAT_ID": "123",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
        "OPENAI_API_KEY": "set",
        "FINNHUB_API_KEY": "set",
        "VERCEL_TOKEN": "set",
    }
    report = build_live_readiness_report(
        project_root=Path("/tmp/apex"),
        settings_raw=_settings(),
        env=env,
        command_exists=lambda command: True,
        git_probe={"is_repo": True, "remote": "git@github.com:ArohaBookings/nexustrader.git", "branch": "main", "upstream": "origin/main"},
        gh_auth_probe={"ok": True},
        vercel_probe={"ok": True, "project_linked": True, "token_present": True},
        mt5_verification={"ok": True, "account": {"login": 1}, "version": "5"},
        bridge_health={
            "ok": True,
            "bridge_status": "UP",
            "broker_connectivity": {
                "terminal_connected": True,
                "terminal_trade_allowed": None,
                "mql_trade_allowed": True,
                "ea_polling_fresh": True,
                "explicit_permission_flags": False,
                "account": "1",
            },
            "latest_account_snapshot": {
                "account": "1",
                "balance": 100.0,
                "equity": 101.0,
                "free_margin": 95.0,
                "bid": 68000.10,
                "ask": 68000.40,
                "last_price": 68000.25,
            },
        },
        telegram_identity={"ok": True, "username": "Nexus_vantage_trader_bot", "is_bot": True},
    )

    assert report["ready"] is True
    assert "bridge_mt5_feed" not in {item["name"] for item in report["hard_blockers"]}
    assert "bridge_mt5_permission_flags" in {item["name"] for item in report["warnings"]}


def test_live_readiness_blocks_missing_broker_price_context() -> None:
    env = {
        "TELEGRAM_BOT_TOKEN": "set",
        "TELEGRAM_CHAT_ID": "123",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
        "OPENAI_API_KEY": "set",
        "FINNHUB_API_KEY": "set",
    }
    report = build_live_readiness_report(
        project_root=Path("/tmp/apex"),
        settings_raw=_settings(),
        env=env,
        command_exists=lambda command: command in {"git", "gh", "vercel"},
        git_probe={"is_repo": True, "remote": "git@github.com:ArohaBookings/nexustrader.git", "branch": "main", "upstream": "origin/main"},
        gh_auth_probe={"ok": True},
        vercel_probe={"ok": True, "project_linked": True, "token_present": True},
        mt5_verification={"ok": True, "account": {"login": 1}, "version": "5"},
        bridge_health={
            "ok": True,
            "bridge_status": "UP",
            "broker_connectivity": {"terminal_connected": True, "ea_polling_fresh": True, "account": "1"},
            "latest_account_snapshot": {"account": "1", "balance": 100.0, "equity": 101.0, "free_margin": 95.0},
        },
        telegram_identity={"ok": True, "username": "Nexus_vantage_trader_bot", "is_bot": True},
    )

    blocker_names = {item["name"] for item in report["hard_blockers"]}
    assert report["ready"] is False
    assert "bridge_broker_price_context" in blocker_names


def test_live_readiness_blocks_zero_equity_bridge_snapshot() -> None:
    env = {
        "TELEGRAM_BOT_TOKEN": "set",
        "TELEGRAM_CHAT_ID": "123",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
        "OPENAI_API_KEY": "set",
        "FINNHUB_API_KEY": "set",
    }
    report = build_live_readiness_report(
        project_root=Path("/tmp/apex"),
        settings_raw=_settings(),
        env=env,
        command_exists=lambda command: command in {"git", "gh", "vercel"},
        git_probe={"is_repo": True, "remote": "git@github.com:ArohaBookings/nexustrader.git", "branch": "main", "upstream": "origin/main"},
        gh_auth_probe={"ok": True},
        vercel_probe={"ok": True, "project_linked": True, "token_present": True},
        mt5_verification={"ok": True, "account": {"login": 1}, "version": "5"},
        bridge_health={
            "ok": True,
            "bridge_status": "UP",
            "broker_connectivity": {"terminal_connected": True, "ea_polling_fresh": True, "account": "1"},
            "latest_account_snapshot": {"account": "1", "balance": 0.0, "equity": 0.0, "free_margin": 0.0},
        },
        telegram_identity={"ok": True, "username": "Nexus_vantage_trader_bot", "is_bot": True},
    )

    blocker_names = {item["name"] for item in report["hard_blockers"]}
    assert report["ready"] is False
    assert "bridge_account_equity" in blocker_names


def test_live_readiness_blocks_missing_runtime_and_credentials() -> None:
    report = build_live_readiness_report(
        project_root=Path("/tmp/apex"),
        settings_raw=_settings(),
        env={},
        command_exists=lambda command: command == "git",
        git_probe={"is_repo": False},
        gh_auth_probe={"ok": False},
        vercel_probe={"ok": False},
        mt5_verification={"ok": False, "reasons": ["mt5_initialize_failed"]},
        bridge_health={"ok": False, "error": "connection refused"},
        telegram_identity={"ok": False, "error": "missing token"},
    )

    blocker_names = {item["name"] for item in report["hard_blockers"]}
    assert report["ready"] is False
    assert "required_env" in blocker_names
    assert "mt5_connection" in blocker_names
    assert "bridge_health" in blocker_names
    assert "telegram_bot_identity" in blocker_names
    assert "github_auth" in blocker_names
    assert "vercel_auth_project" in blocker_names
    assert any("send /start" in action for action in report["next_actions"])


def test_live_readiness_blocks_dirty_git_worktree_for_deploy() -> None:
    env = {
        "TELEGRAM_BOT_TOKEN": "set",
        "TELEGRAM_CHAT_ID": "123",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
        "OPENAI_API_KEY": "set",
        "FINNHUB_API_KEY": "set",
    }
    report = build_live_readiness_report(
        project_root=Path("/tmp/apex"),
        settings_raw=_settings(),
        env=env,
        command_exists=lambda command: True,
        git_probe={
            "is_repo": True,
            "remote": "https://github.com/ArohaBookings/nexustrader.git",
            "branch": "main",
            "upstream": "origin/main",
            "has_head": True,
            "dirty_count": 2,
            "dirty_preview": [" M src/live_readiness.py", "?? scratch.txt"],
        },
        gh_auth_probe={"ok": True},
        vercel_probe={"ok": True, "project_linked": True, "token_present": True},
        mt5_verification={"ok": True, "account": {"login": 1}, "version": "5"},
        bridge_health={
            "ok": True,
            "bridge_status": "UP",
            "broker_connectivity": {"terminal_connected": True, "terminal_trade_allowed": True},
        },
        telegram_identity={"ok": True, "username": "Nexus_vantage_trader_bot", "is_bot": True},
    )

    blocker_names = {item["name"] for item in report["hard_blockers"]}
    assert "git_worktree_clean" in blocker_names
    assert report["ready"] is False


def test_live_readiness_blocks_unpublished_git_branch_for_deploy() -> None:
    env = {
        "TELEGRAM_BOT_TOKEN": "set",
        "TELEGRAM_CHAT_ID": "123",
        "TELEGRAM_WEBHOOK_SECRET": "secret",
        "OPENAI_API_KEY": "set",
        "FINNHUB_API_KEY": "set",
    }
    report = build_live_readiness_report(
        project_root=Path("/tmp/apex"),
        settings_raw=_settings(),
        env=env,
        command_exists=lambda command: True,
        git_probe={
            "is_repo": True,
            "remote": "https://github.com/ArohaBookings/nexustrader.git",
            "branch": "main",
            "upstream": "",
            "has_head": True,
            "dirty_count": 0,
        },
        gh_auth_probe={"ok": True},
        vercel_probe={"ok": True, "project_linked": True, "token_present": True},
        mt5_verification={"ok": True, "account": {"login": 1}, "version": "5"},
        bridge_health={
            "ok": True,
            "bridge_status": "UP",
            "broker_connectivity": {"terminal_connected": True, "terminal_trade_allowed": True},
        },
        telegram_identity={"ok": True, "username": "Nexus_vantage_trader_bot", "is_bot": True},
    )

    blocker_names = {item["name"] for item in report["hard_blockers"]}
    assert "git_branch_published" in blocker_names
    assert report["ready"] is False
