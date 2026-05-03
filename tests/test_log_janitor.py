from __future__ import annotations

from datetime import datetime, timezone
import gzip
import json
import os

from src.log_janitor import LogJanitor


def test_log_janitor_compresses_old_logs_skips_active_and_preserves_learning(tmp_path) -> None:
    logs_dir = tmp_path / "logs"
    data_dir = tmp_path / "data"
    logs_dir.mkdir()
    data_dir.mkdir()
    old_log = logs_dir / "old_strategy.log"
    recent_log = logs_dir / "recent_strategy.log"
    active_log = logs_dir / "local_bridge.log"
    old_jsonl = data_dir / "old_events.jsonl"
    old_log.write_text("TRADE_OPEN XAUUSD\nREJECT invalid_stops\nLEARNING promoted\n", encoding="utf-8")
    recent_log.write_text("TRADE_RECENT\n", encoding="utf-8")
    active_log.write_text("TRADE_ACTIVE\n", encoding="utf-8")
    old_jsonl.write_text(json.dumps({"level": "INFO", "message": "TRADE_CLOSE winner"}) + "\n", encoding="utf-8")
    old_ts = datetime(2026, 4, 1, tzinfo=timezone.utc).timestamp()
    recent_ts = datetime(2026, 5, 3, tzinfo=timezone.utc).timestamp()
    for path in (old_log, old_jsonl, active_log):
        os.utime(path, (old_ts, old_ts))
    os.utime(recent_log, (recent_ts, recent_ts))

    janitor = LogJanitor.from_mapping(
        project_root=tmp_path,
        logs_dir=logs_dir,
        data_dir=data_dir,
        raw={
            "enabled": True,
            "compress_after_days": 14,
            "compressed_retention_days": 365,
            "preserve_learning_memory": True,
            "audit_file": str(data_dir / "janitor_runs.jsonl"),
            "learning_summary_file": str(data_dir / "learning_memory" / "log_janitor_summaries.jsonl"),
        },
    )

    result = janitor.run(now=datetime(2026, 5, 4, tzinfo=timezone.utc))

    assert result["ok"] is True
    assert result["compressed"] == 2
    assert (logs_dir / "old_strategy.log.gz").exists()
    assert (data_dir / "old_events.jsonl.gz").exists()
    assert recent_log.exists()
    assert active_log.exists()
    with gzip.open(logs_dir / "old_strategy.log.gz", "rt", encoding="utf-8") as handle:
        assert "TRADE_OPEN" in handle.read()

    summaries = (data_dir / "learning_memory" / "log_janitor_summaries.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(summaries) == 2
    assert any("TRADE_" in line or "REJECT" in line for line in summaries)
    audit = (data_dir / "janitor_runs.jsonl").read_text(encoding="utf-8")
    assert '"compressed": 2' in audit
