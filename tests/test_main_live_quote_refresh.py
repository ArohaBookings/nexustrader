from __future__ import annotations

from datetime import datetime, timezone
import unittest

import pandas as pd

from src.main import _refresh_frame_with_bridge_quote


class MainLiveQuoteRefreshTests(unittest.TestCase):
    def test_refresh_frame_with_bridge_quote_appends_current_bucket(self) -> None:
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2026-03-27T05:15:00Z",
                        "2026-03-27T05:20:00Z",
                    ],
                    utc=True,
                ),
                "open": [4438.2, 4438.2],
                "high": [4438.9, 4438.6],
                "low": [4437.8, 4437.4],
                "close": [4438.2, 4437.5],
                "spread": [16.0, 16.0],
                "tick_volume": [12, 10],
                "real_volume": [12, 10],
            }
        )
        snapshot = {
            "last": 4441.1,
            "bid": 4441.0,
            "ask": 4441.2,
            "spread_points": 14.0,
            "updated_at": "2026-03-27T05:34:44+00:00",
        }

        refreshed = _refresh_frame_with_bridge_quote(
            frame,
            timeframe="M5",
            bridge_symbol_snapshot=snapshot,
            now_utc=datetime(2026, 3, 27, 5, 34, 50, tzinfo=timezone.utc),
        )

        self.assertIsNotNone(refreshed)
        assert refreshed is not None
        self.assertEqual(len(refreshed), 3)
        self.assertEqual(pd.Timestamp(refreshed.iloc[-1]["time"]), pd.Timestamp("2026-03-27T05:30:00Z"))
        self.assertAlmostEqual(float(refreshed.iloc[-1]["open"]), 4437.5)
        self.assertAlmostEqual(float(refreshed.iloc[-1]["close"]), 4441.1)
        self.assertAlmostEqual(float(refreshed.iloc[-1]["high"]), 4441.1)
        self.assertAlmostEqual(float(refreshed.iloc[-1]["low"]), 4437.5)
