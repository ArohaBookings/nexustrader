from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
import json
import logging
from logging.handlers import RotatingFileHandler
import sys
import time

from src.utils import ensure_parent, utc_now


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": utc_now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ApexLogger(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        extra_fields = dict(self.extra)
        extra_fields.update(extra.get("extra_fields", {}))
        extra["extra_fields"] = extra_fields
        return msg, kwargs

    def bind(self, **fields: Any) -> "ApexLogger":
        merged = dict(self.extra)
        merged.update(fields)
        return ApexLogger(self.logger, merged)


@dataclass
class LoggerFactory:
    name: str = "apex"
    log_file: Path | None = None
    rotate_max_bytes: int = 10 * 1024 * 1024
    rotate_backup_count: int = 7
    retention_days: int = 7

    def build(self) -> ApexLogger:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        for existing in list(logger.handlers):
            try:
                existing.flush()
                existing.close()
            except Exception:
                pass
            logger.removeHandler(existing)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(JsonFormatter())
        logger.addHandler(stream_handler)

        if self.log_file:
            ensure_parent(self.log_file)
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=max(1, int(self.rotate_max_bytes)),
                backupCount=max(1, int(self.rotate_backup_count)),
            )
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)
            self._purge_old_logs()

        logger.propagate = False
        return ApexLogger(logger, {})

    def _purge_old_logs(self) -> None:
        if not self.log_file:
            return
        cutoff = time.time() - max(1, int(self.retention_days)) * int(timedelta(days=1).total_seconds())
        pattern = f"{self.log_file.name}*"
        for candidate in self.log_file.parent.glob(pattern):
            if not candidate.is_file():
                continue
            try:
                if candidate.stat().st_mtime < cutoff:
                    candidate.unlink(missing_ok=True)
            except Exception:
                continue
