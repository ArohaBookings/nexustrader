#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

UID_NUM="$(id -u)"
LAUNCH_DIR="${HOME}/Library/LaunchAgents"

for label in "com.apexbot.telegram" "com.apexbot.bridge"; do
  plist="${LAUNCH_DIR}/${label}.plist"
  if [[ -f "${plist}" ]]; then
    launchctl bootout "gui/${UID_NUM}" "${plist}" >/dev/null 2>&1 || true
    echo "${label}: launch agent stopped"
  fi
done

for session in "${APEX_TELEGRAM_SCREEN_SESSION:-apex_telegram}" "${APEX_BRIDGE_SCREEN_SESSION:-apex_bridge}"; do
  if command -v screen >/dev/null 2>&1; then
    if screen -S "${session}" -X quit >/dev/null 2>&1; then
      echo "${session}: screen session stopped"
    fi
    screen -wipe >/dev/null 2>&1 || true
  fi
done

pkill -f "scripts/apex_telegram_poll.py --claim-owner" >/dev/null 2>&1 || true
pkill -f -- "-m src.main --bridge-only" >/dev/null 2>&1 || true

wait_for_no_process() {
  local label="$1"
  local pattern="$2"
  local i
  for ((i = 1; i <= 20; i++)); do
    if ! pgrep -f -- "${pattern}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  pkill -9 -f -- "${pattern}" >/dev/null 2>&1 || true
  echo "${label}: forced stale process cleanup"
}

wait_for_bridge_down() {
  local i
  for ((i = 1; i <= 20; i++)); do
    if ! python3 - <<'PY' >/dev/null 2>&1
from urllib.request import urlopen
urlopen("http://127.0.0.1:8000/health", timeout=1).read()
PY
    then
      return 0
    fi
    sleep 0.25
  done
  echo "APEX bridge: health endpoint still responding after stop request"
}

wait_for_no_process "APEX Telegram poller" "scripts/apex_telegram_poll.py --claim-owner"
wait_for_no_process "APEX bridge" "-m src.main --bridge-only"
wait_for_bridge_down

stop_pid_file() {
  local label="$1"
  local pid_file="$2"
  if [[ ! -f "${pid_file}" ]]; then
    echo "${label}: no pid file"
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
    rm -f "${pid_file}"
    echo "${label}: stale pid file removed"
    return 0
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    echo "${label}: stopped pid ${pid}"
  else
    echo "${label}: pid ${pid} was not running"
  fi
  rm -f "${pid_file}"
}

stop_pid_file "APEX Telegram poller" "data/run/apex_telegram_poll.pid"
stop_pid_file "APEX bridge" "data/run/apex_bridge.pid"
