from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import gzip
import json
import os
import shutil
import time

UTC = timezone.utc


@dataclass(frozen=True)
class LogJanitorConfig:
    enabled: bool = True
    compress_after_days: int = 14
    compressed_retention_days: int = 365
    preserve_learning_memory: bool = True
    audit_file: str = "data/janitor_runs.jsonl"
    learning_summary_file: str = "data/learning_memory/log_janitor_summaries.jsonl"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "LogJanitorConfig":
        data = dict(raw or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            compress_after_days=max(1, int(_number(data.get("compress_after_days"), 14))),
            compressed_retention_days=max(1, int(_number(data.get("compressed_retention_days"), 365))),
            preserve_learning_memory=bool(data.get("preserve_learning_memory", True)),
            audit_file=str(data.get("audit_file") or "data/janitor_runs.jsonl"),
            learning_summary_file=str(data.get("learning_summary_file") or "data/learning_memory/log_janitor_summaries.jsonl"),
        )


class LogJanitor:
    def __init__(
        self,
        *,
        project_root: Path,
        logs_dir: Path,
        data_dir: Path,
        config: LogJanitorConfig,
    ) -> None:
        self.project_root = Path(project_root)
        self.logs_dir = Path(logs_dir)
        self.data_dir = Path(data_dir)
        self.config = config
        self.audit_file = self._resolve(config.audit_file)
        self.learning_summary_file = self._resolve(config.learning_summary_file)

    @classmethod
    def from_mapping(
        cls,
        *,
        project_root: Path,
        logs_dir: Path,
        data_dir: Path,
        raw: Mapping[str, Any] | None,
    ) -> "LogJanitor":
        return cls(
            project_root=project_root,
            logs_dir=logs_dir,
            data_dir=data_dir,
            config=LogJanitorConfig.from_mapping(raw),
        )

    def run(self, *, now: datetime | None = None) -> dict[str, Any]:
        current = now.astimezone(UTC) if now and now.tzinfo else (now.replace(tzinfo=UTC) if now else datetime.now(tz=UTC))
        if not self.config.enabled:
            result = {"ok": True, "enabled": False, "compressed": 0, "deleted": 0, "skipped": 0, "ran_at": current.isoformat()}
            self._audit(result)
            return result
        cutoff = current.timestamp() - (self.config.compress_after_days * 86400)
        retention_cutoff = current.timestamp() - (self.config.compressed_retention_days * 86400)
        compressed: list[str] = []
        deleted: list[str] = []
        skipped = 0
        errors: list[dict[str, str]] = []
        active_names = self._active_log_names()
        for path in self._candidate_files():
            try:
                if path.name in active_names:
                    skipped += 1
                    continue
                stat = path.stat()
                if path.suffix == ".gz":
                    if stat.st_mtime < retention_cutoff:
                        path.unlink(missing_ok=True)
                        deleted.append(str(path))
                    continue
                if stat.st_mtime >= cutoff:
                    skipped += 1
                    continue
                if self.config.preserve_learning_memory:
                    self._preserve_summary(path, current)
                gz_path = path.with_name(path.name + ".gz")
                self._gzip_file(path, gz_path)
                compressed.append(str(gz_path))
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})
        result = {
            "ok": not errors,
            "enabled": True,
            "ran_at": current.isoformat(),
            "compress_after_days": int(self.config.compress_after_days),
            "compressed_retention_days": int(self.config.compressed_retention_days),
            "compressed": len(compressed),
            "deleted": len(deleted),
            "skipped": int(skipped),
            "compressed_files": compressed[:50],
            "deleted_files": deleted[:50],
            "errors": errors[:20],
        }
        self._audit(result)
        return result

    def _candidate_files(self) -> Iterable[Path]:
        seen: set[Path] = set()
        roots = [self.logs_dir, self.data_dir]
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            for pattern in ("*.log", "*.log.*", "*.jsonl", "*.jsonl.*", "*.gz"):
                for path in root.glob(pattern):
                    resolved = path.resolve()
                    if resolved in seen or not path.is_file():
                        continue
                    seen.add(resolved)
                    yield path

    def _active_log_names(self) -> set[str]:
        return {
            "local_bridge.log",
            "telegram_poll.log",
            "nexus_collector.log",
            "apex.log",
        }

    def _preserve_summary(self, path: Path, current: datetime) -> None:
        summary = {
            "source_file": str(path),
            "compressed_at": current.isoformat(),
            "bytes": int(path.stat().st_size),
            "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
            "signals": self._extract_signals(path),
        }
        self.learning_summary_file.parent.mkdir(parents=True, exist_ok=True)
        with self.learning_summary_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True, default=str) + "\n")

    def _extract_signals(self, path: Path) -> dict[str, Any]:
        counts: dict[str, int] = {}
        levels: dict[str, int] = {}
        sample_messages: list[str] = []
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for idx, line in enumerate(handle):
                    if idx >= 5000:
                        break
                    text = line.strip()
                    if not text:
                        continue
                    message = text[:180]
                    level = ""
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, Mapping):
                            message = str(parsed.get("message") or message)[:180]
                            level = str(parsed.get("level") or "").upper()
                    except Exception:
                        pass
                    if level:
                        levels[level] = levels.get(level, 0) + 1
                    for token in ("TRADE_", "GRID_", "BLOCK", "REJECT", "ERROR", "WARNING", "LEARNING", "PROMOT"):
                        if token.lower() in message.lower():
                            counts[token] = counts.get(token, 0) + 1
                    if len(sample_messages) < 5 and any(token.lower() in message.lower() for token in ("trade", "reject", "block", "learning", "promot")):
                        sample_messages.append(message)
        except OSError:
            pass
        return {"counts": counts, "levels": levels, "samples": sample_messages}

    def _gzip_file(self, source: Path, target: Path) -> None:
        with source.open("rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        os.utime(target, (source.stat().st_atime, source.stat().st_mtime))
        source.unlink()

    def _audit(self, payload: Mapping[str, Any]) -> None:
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")

    def _resolve(self, value: str) -> Path:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
