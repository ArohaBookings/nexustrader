from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from src.openai_client import OpenAIClient


class OpenAIClientTests(TestCase):
    def test_request_uses_total_timeout_budget_across_retry(self) -> None:
        client = OpenAIClient(enabled=True, timeout_seconds=2.0, retry_once=True)
        seen_timeouts: list[float] = []

        def fake_post_json(_url: str, _payload: dict, timeout_seconds: float | None = None) -> dict:
            seen_timeouts.append(float(timeout_seconds or 0.0))
            raise TimeoutError()

        client._post_json = fake_post_json  # type: ignore[method-assign]

        with patch("src.openai_client.time.monotonic", side_effect=[0.0, 0.0, 0.1, 1.6, 1.7]):
            payload, error = client._request({"ping": True})

        self.assertIsNone(payload)
        self.assertEqual(error, "openai_timeout")
        self.assertEqual(len(seen_timeouts), 2)
        self.assertAlmostEqual(seen_timeouts[0], 2.0, places=2)
        self.assertLessEqual(seen_timeouts[1], 0.41)

    def test_request_skips_retry_when_timeout_budget_is_exhausted(self) -> None:
        client = OpenAIClient(enabled=True, timeout_seconds=2.0, retry_once=True)
        call_count = 0

        def fake_post_json(_url: str, _payload: dict, timeout_seconds: float | None = None) -> dict:
            nonlocal call_count
            call_count += 1
            raise TimeoutError()

        client._post_json = fake_post_json  # type: ignore[method-assign]

        with patch("src.openai_client.time.monotonic", side_effect=[0.0, 0.0, 0.1, 2.1]):
            payload, error = client._request({"ping": True})

        self.assertIsNone(payload)
        self.assertEqual(error, "openai_timeout")
        self.assertEqual(call_count, 1)

    def test_request_respects_timeout_cooldown_after_failure(self) -> None:
        client = OpenAIClient(enabled=True, timeout_seconds=2.0, retry_once=False)
        call_count = 0

        def fake_post_json(_url: str, _payload: dict, timeout_seconds: float | None = None) -> dict:
            nonlocal call_count
            call_count += 1
            raise TimeoutError()

        client._post_json = fake_post_json  # type: ignore[method-assign]

        with patch("src.openai_client.time.monotonic", side_effect=[0.0, 0.0, 0.1, 0.5]):
            payload1, error1 = client._request({"ping": True})
            payload2, error2 = client._request({"ping": True})

        self.assertIsNone(payload1)
        self.assertEqual(error1, "openai_timeout")
        self.assertIsNone(payload2)
        self.assertEqual(error2, "openai_cooldown_active")
        self.assertEqual(call_count, 1)
