from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os


@dataclass(frozen=True)
class EnvLoadResult:
    files_loaded: tuple[str, ...]
    keys_loaded: tuple[str, ...]

    @property
    def loaded_count(self) -> int:
        return len(self.keys_loaded)


def load_env_files(root: Path, logger: Any | None = None) -> EnvLoadResult:
    candidates = [root / "config" / "secrets.env", root / "secrets.env"]
    loaded_files: list[str] = []
    loaded_keys: list[str] = []

    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            env_key = key.strip()
            if not env_key:
                continue
            if env_key in os.environ:
                continue
            os.environ[env_key] = value.strip()
            loaded_keys.append(env_key)
        loaded_files.append(str(path))

    result = EnvLoadResult(files_loaded=tuple(loaded_files), keys_loaded=tuple(sorted(set(loaded_keys))))
    if logger is not None and hasattr(logger, "info"):
        logger.info(
            "env_files_loaded",
            extra={
                "extra_fields": {
                    "files": list(result.files_loaded),
                    "keys_loaded_count": result.loaded_count,
                    "keys": list(result.keys_loaded),
                }
            },
        )
    return result
