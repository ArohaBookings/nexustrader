from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest

from src.env_loader import load_env_files


class _Logger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict]] = []

    def info(self, message: str, extra: dict | None = None) -> None:
        self.messages.append((message, extra or {}))


class EnvLoaderTests(unittest.TestCase):
    def test_loads_config_and_root_env_files_without_overriding_existing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "secrets.env").write_text("OPENAI_API_KEY=config-key\nNEWS_API_KEY=config-news\n")
            (root / "secrets.env").write_text("BRIDGE_AUTH_TOKEN=root-token\nOPENAI_API_KEY=root-key\n")

            original_openai = os.environ.get("OPENAI_API_KEY")
            original_news = os.environ.get("NEWS_API_KEY")
            original_bridge = os.environ.get("BRIDGE_AUTH_TOKEN")
            os.environ["OPENAI_API_KEY"] = "already-set"
            try:
                result = load_env_files(root)
                self.assertEqual(os.environ.get("OPENAI_API_KEY"), "already-set")
            finally:
                if original_openai is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = original_openai
                if original_news is None:
                    os.environ.pop("NEWS_API_KEY", None)
                else:
                    os.environ["NEWS_API_KEY"] = original_news
                if original_bridge is None:
                    os.environ.pop("BRIDGE_AUTH_TOKEN", None)
                else:
                    os.environ["BRIDGE_AUTH_TOKEN"] = original_bridge

            self.assertIn("NEWS_API_KEY", result.keys_loaded)
            self.assertIn("BRIDGE_AUTH_TOKEN", result.keys_loaded)
            self.assertNotIn("OPENAI_API_KEY", result.keys_loaded)

    def test_logger_does_not_emit_secret_values(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            secret_value = "super-secret-value"
            (root / "config" / "secrets.env").write_text(f"OPENAI_API_KEY={secret_value}\n")
            logger = _Logger()
            original = os.environ.get("OPENAI_API_KEY")
            try:
                os.environ.pop("OPENAI_API_KEY", None)
                result = load_env_files(root, logger=logger)
            finally:
                if original is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = original

            self.assertGreaterEqual(result.loaded_count, 1)
            flattened = str(logger.messages)
            self.assertNotIn(secret_value, flattened)


if __name__ == "__main__":
    unittest.main()
