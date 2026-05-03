from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os


def _env_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    runtime_root: Path
    config_dir: Path
    data_dir: Path
    cache_dir: Path
    logs_dir: Path
    state_dir: Path
    models_dir: Path
    temp_dir: Path

    def resolve(self, value: str) -> Path:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path.resolve()
        normalized = str(value).replace("\\", "/")
        if normalized.startswith("config/"):
            return (self.config_dir / normalized.removeprefix("config/")).resolve()
        if normalized.startswith("models/"):
            return (self.models_dir / normalized.removeprefix("models/")).resolve()
        if normalized.startswith("data/"):
            relative = normalized.removeprefix("data/")
            if relative.startswith("candles_cache") or relative.startswith("news_cache"):
                return (self.cache_dir / relative).resolve()
            if relative.startswith("alerts") or relative.startswith("apex.log"):
                return (self.logs_dir / relative).resolve()
            return (self.state_dir / relative).resolve()
        if normalized.startswith("tmp/") or normalized.startswith("temp/"):
            return (self.temp_dir / normalized.split("/", 1)[1]).resolve()
        return (self.runtime_root / normalized).resolve()

    def snapshot(self) -> dict[str, str]:
        return {
            "project_root": str(self.project_root),
            "runtime_root": str(self.runtime_root),
            "config_dir": str(self.config_dir),
            "data_dir": str(self.data_dir),
            "cache_dir": str(self.cache_dir),
            "logs_dir": str(self.logs_dir),
            "state_dir": str(self.state_dir),
            "models_dir": str(self.models_dir),
            "temp_dir": str(self.temp_dir),
        }


def resolve_runtime_paths(project_root: Path, raw: dict[str, Any]) -> RuntimePaths:
    runtime_root = _env_path("APEX_RUNTIME_ROOT") or project_root
    config_dir = _env_path("APEX_CONFIG_DIR") or (project_root / "config").resolve()
    data_dir = _env_path("APEX_DATA_DIR") or (runtime_root / "data").resolve()
    cache_dir = _env_path("APEX_CACHE_DIR") or data_dir
    logs_dir = _env_path("APEX_LOGS_DIR") or data_dir
    state_dir = _env_path("APEX_STATE_DIR") or data_dir
    models_dir = _env_path("APEX_MODELS_DIR") or (runtime_root / "models").resolve()
    temp_dir = _env_path("APEX_TEMP_DIR") or (runtime_root / "tmp").resolve()

    paths = RuntimePaths(
        project_root=project_root.resolve(),
        runtime_root=runtime_root.resolve(),
        config_dir=config_dir.resolve(),
        data_dir=data_dir.resolve(),
        cache_dir=cache_dir.resolve(),
        logs_dir=logs_dir.resolve(),
        state_dir=state_dir.resolve(),
        models_dir=models_dir.resolve(),
        temp_dir=temp_dir.resolve(),
    )

    raw["runtime_paths"] = paths.snapshot()
    return paths
