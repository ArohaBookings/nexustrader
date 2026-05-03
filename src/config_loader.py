from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import yaml

from src.env_loader import load_env_files
from src.runtime_paths import RuntimePaths, resolve_runtime_paths
from src.utils import SessionWindow, ensure_directory, parse_hhmm


@dataclass
class Settings:
    root: Path
    runtime_paths: RuntimePaths
    raw: dict[str, Any]
    sessions: list[SessionWindow]

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"Config section {name} must be a mapping")
        return value

    def path(self, dotted_key: str) -> Path:
        current: Any = self.raw
        for key in dotted_key.split("."):
            if not isinstance(current, dict):
                raise KeyError(dotted_key)
            current = current[key]
        if not isinstance(current, str):
            raise ValueError(f"Config value {dotted_key} is not a string path")
        return self.runtime_paths.resolve(current)

    def resolve_path_value(self, value: str) -> Path:
        return self.runtime_paths.resolve(value)

    def symbols(self) -> list[str]:
        system = self.section("system")
        raw_symbols = system.get("symbols")
        if isinstance(raw_symbols, list):
            symbols = [str(symbol).upper() for symbol in raw_symbols if str(symbol).strip()]
            if symbols:
                return symbols
        fallback = str(system.get("symbol", "XAUUSD")).upper()
        return [fallback]

    def primary_symbol(self) -> str:
        return self.symbols()[0]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _apply_env_overrides(raw: dict[str, Any]) -> None:
    system = raw.setdefault("system", {})
    risk = raw.setdefault("risk", {})
    bridge = raw.setdefault("bridge", {})
    dashboard = raw.setdefault("dashboard", {})
    if "APEX_MODE" in os.environ:
        system["mode"] = os.environ["APEX_MODE"].upper()
    if "APEX_SYMBOL" in os.environ:
        system["symbol"] = os.environ["APEX_SYMBOL"].upper()
    if "APEX_SYMBOLS" in os.environ:
        system["symbols"] = [symbol.strip().upper() for symbol in os.environ["APEX_SYMBOLS"].split(",") if symbol.strip()]
    if "LIVE_TRADING" in os.environ:
        system["live_trading"] = os.environ["LIVE_TRADING"].strip().lower() in {"1", "true", "yes", "on"}
    if "APEX_MAX_POSITIONS" in os.environ:
        max_positions = max(1, int(os.environ["APEX_MAX_POSITIONS"]))
        system["max_positions"] = max_positions
        system["max_positions_total"] = max_positions
        risk["max_positions_total"] = max_positions
    if "APEX_RISK_PER_TRADE" in os.environ:
        risk["risk_per_trade"] = float(os.environ["APEX_RISK_PER_TRADE"])
    if "BRIDGE_HOST" in os.environ:
        bridge["host"] = os.environ["BRIDGE_HOST"].strip()
    if "BRIDGE_PORT" in os.environ:
        bridge["port"] = int(os.environ["BRIDGE_PORT"])
    if "BRIDGE_AUTH_TOKEN" in os.environ:
        bridge["auth_token"] = os.environ["BRIDGE_AUTH_TOKEN"].strip()
    if "APEX_DASHBOARD_ENABLED" in os.environ:
        dashboard["enabled"] = os.environ["APEX_DASHBOARD_ENABLED"].strip().lower() in {"1", "true", "yes", "on"}
    if "APEX_DASHBOARD_PUBLIC_ENABLED" in os.environ:
        dashboard["public_enabled"] = os.environ["APEX_DASHBOARD_PUBLIC_ENABLED"].strip().lower() in {"1", "true", "yes", "on"}
    if "APEX_DASHBOARD_ALLOW_PUBLIC_HEALTH" in os.environ:
        dashboard["allow_public_health"] = os.environ["APEX_DASHBOARD_ALLOW_PUBLIC_HEALTH"].strip().lower() in {"1", "true", "yes", "on"}
    if "APEX_DASHBOARD_PASSWORD" in os.environ:
        dashboard["password"] = os.environ["APEX_DASHBOARD_PASSWORD"].strip()
    elif "DASHBOARD_PASSWORD" in os.environ:
        dashboard["password"] = os.environ["DASHBOARD_PASSWORD"].strip()
    if "APEX_DASHBOARD_SESSION_SECRET" in os.environ:
        dashboard["session_secret"] = os.environ["APEX_DASHBOARD_SESSION_SECRET"].strip()
    elif "DASHBOARD_SESSION_SECRET" in os.environ:
        dashboard["session_secret"] = os.environ["DASHBOARD_SESSION_SECRET"].strip()
    if "APEX_DASHBOARD_SESSION_TIMEOUT_MINUTES" in os.environ:
        dashboard["session_timeout_minutes"] = int(os.environ["APEX_DASHBOARD_SESSION_TIMEOUT_MINUTES"])
    if "APEX_DASHBOARD_READ_ONLY" in os.environ:
        dashboard["read_only"] = os.environ["APEX_DASHBOARD_READ_ONLY"].strip().lower() in {"1", "true", "yes", "on"}
    if "APEX_DASHBOARD_ALLOWED_IPS" in os.environ:
        dashboard["allowed_ips"] = [item.strip() for item in os.environ["APEX_DASHBOARD_ALLOWED_IPS"].split(",") if item.strip()]
    if "APEX_DASHBOARD_MOBILE_REFRESH_SECONDS" in os.environ:
        dashboard["mobile_refresh_seconds"] = int(os.environ["APEX_DASHBOARD_MOBILE_REFRESH_SECONDS"])
    if "APEX_DASHBOARD_DESKTOP_REFRESH_SECONDS" in os.environ:
        dashboard["desktop_refresh_seconds"] = int(os.environ["APEX_DASHBOARD_DESKTOP_REFRESH_SECONDS"])
    if "APEX_DASHBOARD_HOST" in os.environ:
        dashboard["host"] = os.environ["APEX_DASHBOARD_HOST"].strip()
    if "APEX_DASHBOARD_PORT" in os.environ:
        dashboard["port"] = int(os.environ["APEX_DASHBOARD_PORT"])
    if "APEX_DASHBOARD_BIND_HOST" in os.environ:
        dashboard["bind_host"] = os.environ["APEX_DASHBOARD_BIND_HOST"].strip()
    if "APEX_DASHBOARD_BIND_PORT" in os.environ:
        dashboard["bind_port"] = int(os.environ["APEX_DASHBOARD_BIND_PORT"])


def _build_sessions(session_yaml: dict[str, Any]) -> list[SessionWindow]:
    sessions: list[SessionWindow] = []
    for name, payload in session_yaml.get("sessions", {}).items():
        if not isinstance(payload, dict):
            continue
        sessions.append(
            SessionWindow(
                name=name,
                start=parse_hhmm(str(payload.get("start", "00:00"))),
                end=parse_hhmm(str(payload.get("end", "23:59"))),
                enabled=bool(payload.get("enabled", True)),
                size_multiplier=float(payload.get("size_multiplier", 1.0)),
            )
        )
    return sessions


def load_settings(root: Path | None = None) -> Settings:
    project_root = (root or Path(__file__).resolve().parents[1]).resolve()
    load_env_files(project_root)
    runtime_paths = resolve_runtime_paths(project_root, {})
    config_dir = runtime_paths.config_dir
    raw = _load_yaml(config_dir / "settings.yaml")
    _apply_env_overrides(raw)
    runtime_paths = resolve_runtime_paths(project_root, raw)
    sessions = _build_sessions(_load_yaml(config_dir / "sessions.yaml"))

    data_paths = raw.get("data", {})
    if isinstance(data_paths, dict):
        for value in data_paths.values():
            if isinstance(value, str):
                target = runtime_paths.resolve(value)
                if target.suffix:
                    target.parent.mkdir(parents=True, exist_ok=True)
                else:
                    ensure_directory(target)

    models = raw.get("models", {})
    if isinstance(models, dict):
        for value in models.values():
            if isinstance(value, str):
                target = runtime_paths.resolve(value)
                if target.suffix:
                    target.parent.mkdir(parents=True, exist_ok=True)
                else:
                    ensure_directory(target)

    for path in (
        runtime_paths.runtime_root,
        runtime_paths.data_dir,
        runtime_paths.cache_dir,
        runtime_paths.logs_dir,
        runtime_paths.state_dir,
        runtime_paths.models_dir,
        runtime_paths.temp_dir,
    ):
        ensure_directory(path)

    return Settings(root=project_root, runtime_paths=runtime_paths, raw=raw, sessions=sessions)
