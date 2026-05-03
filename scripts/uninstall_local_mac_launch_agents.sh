#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

UID_NUM="$(id -u)"
LAUNCH_DIR="${HOME}/Library/LaunchAgents"
BRIDGE_LABEL="com.apexbot.bridge"
TELEGRAM_LABEL="com.apexbot.telegram"

for label in "${TELEGRAM_LABEL}" "${BRIDGE_LABEL}"; do
  plist="${LAUNCH_DIR}/${label}.plist"
  launchctl bootout "gui/${UID_NUM}" "${plist}" >/dev/null 2>&1 || true
  rm -f "${plist}"
  echo "Removed ${label}"
done
