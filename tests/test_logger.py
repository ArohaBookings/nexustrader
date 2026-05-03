from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.logger import LoggerFactory


class LoggerRotationTests(unittest.TestCase):
    def test_rotating_file_handler_creates_rollover_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "apex.log"
            logger = LoggerFactory(
                log_file=log_path,
                rotate_max_bytes=512,
                rotate_backup_count=2,
            ).build()
            payload = "x" * 256
            for _ in range(20):
                logger.info(payload)
            log_files = sorted(path.name for path in Path(tmp_dir).glob("apex.log*"))
            self.assertTrue(any(name != "apex.log" for name in log_files), msg=f"log files={log_files}")


if __name__ == "__main__":
    unittest.main()

