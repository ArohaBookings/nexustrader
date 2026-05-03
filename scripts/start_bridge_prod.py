#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_verify(env: dict[str, str]) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--verify"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    return result.returncode == 0


def main() -> int:
    child: subprocess.Popen[str] | None = None
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "."

        boot_wait = max(0, int(env.get("APEX_MT5_BOOT_WAIT_SECONDS", "20")))
        verify_retries = max(0, int(env.get("APEX_MT5_VERIFY_RETRIES", "12")))
        verify_sleep = max(1, int(env.get("APEX_MT5_VERIFY_SLEEP_SECONDS", "5")))
        restart_sleep = max(1, int(env.get("APEX_BRIDGE_RESTART_SLEEP_SECONDS", "10")))
        max_restarts = max(0, int(env.get("APEX_BRIDGE_MAX_RESTARTS", "50")))
        skip_verify = env.get("APEX_SKIP_VERIFY", "0").strip() == "1"

        if boot_wait > 0:
            print(f"[APEX] waiting {boot_wait}s for MT5 terminal startup")
            time.sleep(boot_wait)

        if not skip_verify:
            verify_ok = False
            for attempt in range(verify_retries + 1):
                verify_ok = _run_verify(env)
                if verify_ok:
                    break
                if attempt < verify_retries:
                    print(f"[APEX] MT5 verify not ready, retrying in {verify_sleep}s ({attempt + 1}/{verify_retries})")
                    time.sleep(verify_sleep)
            if not verify_ok:
                print("[APEX] verify never turned green; starting bridge anyway so it can reconcile when MT5 comes up")

        command = [sys.executable, "-m", "src.main", "--bridge-serve"]
        restart_count = 0
        while True:
            print(f"[APEX] starting bridge serve (restart #{restart_count})")
            popen_kwargs: dict[str, object] = {"cwd": PROJECT_ROOT, "env": env}
            if os.name == "nt":
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True
            child = subprocess.Popen(command, **popen_kwargs)
            code = child.wait()
            child = None
            if code == 0:
                return 0
            restart_count += 1
            if restart_count > max_restarts:
                print(f"[APEX] bridge exited with code {code}; max restarts reached", file=sys.stderr)
                return code
            print(f"[APEX] bridge exited with code {code}; restarting in {restart_sleep}s")
            time.sleep(restart_sleep)
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            try:
                child.terminate()
                child.wait(timeout=10)
            except Exception:
                try:
                    child.kill()
                except Exception:
                    pass
        print("[APEX] shutdown requested")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
