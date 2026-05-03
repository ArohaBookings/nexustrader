from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest

from src.runtime_paths import resolve_runtime_paths


class RuntimePathsTests(unittest.TestCase):
    def test_runtime_paths_honor_environment_overrides(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "project"
            project_root.mkdir(parents=True, exist_ok=True)
            runtime_root = Path(tmp_dir) / "runtime"
            logs_dir = runtime_root / "custom-logs"

            original_runtime_root = os.environ.get("APEX_RUNTIME_ROOT")
            original_logs_dir = os.environ.get("APEX_LOGS_DIR")
            try:
                os.environ["APEX_RUNTIME_ROOT"] = str(runtime_root)
                os.environ["APEX_LOGS_DIR"] = str(logs_dir)

                paths = resolve_runtime_paths(project_root, {})
            finally:
                if original_runtime_root is None:
                    os.environ.pop("APEX_RUNTIME_ROOT", None)
                else:
                    os.environ["APEX_RUNTIME_ROOT"] = original_runtime_root
                if original_logs_dir is None:
                    os.environ.pop("APEX_LOGS_DIR", None)
                else:
                    os.environ["APEX_LOGS_DIR"] = original_logs_dir

            self.assertEqual(paths.runtime_root, runtime_root.resolve())
            self.assertEqual(paths.logs_dir, logs_dir.resolve())
            self.assertEqual(paths.resolve("config/settings.yaml"), paths.config_dir / "settings.yaml")


if __name__ == "__main__":
    unittest.main()
