#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="$(pwd -P)"
UID_NUM="$(id -u)"
LAUNCH_DIR="${HOME}/Library/LaunchAgents"
BRIDGE_LABEL="com.apexbot.bridge"
TELEGRAM_LABEL="com.apexbot.telegram"
BRIDGE_PLIST="${LAUNCH_DIR}/${BRIDGE_LABEL}.plist"
TELEGRAM_PLIST="${LAUNCH_DIR}/${TELEGRAM_LABEL}.plist"

mkdir -p "${LAUNCH_DIR}" "${ROOT}/logs" "${ROOT}/data/run"

write_plist() {
  local path="$1"
  local label="$2"
  local runner="$3"
  local log_file="$4"
  python3 - "$path" "$label" "$runner" "$ROOT" "$log_file" <<'PY'
import plistlib
import sys

path, label, runner, root, log_file = sys.argv[1:6]
plist = {
    "Label": label,
    "ProgramArguments": ["/bin/bash", runner],
    "WorkingDirectory": root,
    "RunAtLoad": True,
    "KeepAlive": True,
    "StandardOutPath": log_file,
    "StandardErrorPath": log_file,
    "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
}
with open(path, "wb") as fh:
    plistlib.dump(plist, fh, sort_keys=False)
PY
}

load_service() {
  local label="$1"
  local plist="$2"
  launchctl bootout "gui/${UID_NUM}" "${plist}" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/${UID_NUM}" "${plist}"
  launchctl enable "gui/${UID_NUM}/${label}" >/dev/null 2>&1 || true
  launchctl kickstart -k "gui/${UID_NUM}/${label}" >/dev/null 2>&1 || true
}

write_plist \
  "${BRIDGE_PLIST}" \
  "${BRIDGE_LABEL}" \
  "${ROOT}/scripts/run_local_bridge_service.sh" \
  "${ROOT}/logs/local_bridge.log"

write_plist \
  "${TELEGRAM_PLIST}" \
  "${TELEGRAM_LABEL}" \
  "${ROOT}/scripts/run_local_telegram_service.sh" \
  "${ROOT}/logs/telegram_poll.log"

load_service "${BRIDGE_LABEL}" "${BRIDGE_PLIST}"
load_service "${TELEGRAM_LABEL}" "${TELEGRAM_PLIST}"

echo "Installed and started APEX local launch agents:"
echo "- ${BRIDGE_LABEL}"
echo "- ${TELEGRAM_LABEL}"
echo "Bridge health: http://127.0.0.1:8000/health"
echo "Logs: ${ROOT}/logs/local_bridge.log and ${ROOT}/logs/telegram_poll.log"
