from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_settings
from src.log_janitor import LogJanitor


def main() -> int:
    settings = load_settings(ROOT)
    janitor = LogJanitor.from_mapping(
        project_root=ROOT,
        logs_dir=settings.runtime_paths.logs_dir,
        data_dir=settings.runtime_paths.state_dir,
        raw=settings.raw.get("janitor", {}) if isinstance(settings.raw.get("janitor"), dict) else {},
    )
    result = janitor.run()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
