#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

load_env_file() {
  local env_file="$1"
  [[ -f "${env_file}" ]] || return 0
  local line key value
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line}" == \#* || "${line}" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    key="${key#"${key%%[![:space:]]*}"}"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -z "${!key+x}" ]]; then
      export "${key}=${value}"
    fi
  done < "${env_file}"
}

for env_file in "config/secrets.env" "secrets.env"; do
  load_env_file "${env_file}"
done

export APEX_MT5_RUNTIME_MODE="${APEX_MT5_RUNTIME_MODE:-EA_BRIDGE}"

missing=()
for key in TELEGRAM_BOT_TOKEN TELEGRAM_WEBHOOK_SECRET; do
  if [[ -z "${!key:-}" ]]; then
    missing+=("${key}")
  fi
done
if (( ${#missing[@]} > 0 )); then
  printf 'Missing required local env keys: %s\n' "${missing[*]}"
  echo "Populate config/secrets.env. Secret values were not printed."
  exit 1
fi

mkdir -p data/run logs

screen_session_alive() {
  local session="$1"
  command -v screen >/dev/null 2>&1 || return 1
  screen -ls 2>/dev/null | grep -Eq "[.]${session}[[:space:]]"
}

wait_for_screen_session() {
  local session="$1"
  local attempts="${2:-10}"
  local delay="${3:-1}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if screen_session_alive "${session}"; then
      return 0
    fi
    sleep "${delay}"
  done
  return 1
}

telegram_poller_alive() {
  if screen_session_alive "${APEX_TELEGRAM_SCREEN_SESSION:-apex_telegram}"; then
    return 0
  fi
  pgrep -f "scripts/apex_telegram_poll.py --claim-owner" >/dev/null 2>&1
}

pid_alive() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

bridge_healthy() {
  python3 - <<'PY' >/dev/null 2>&1
from urllib.request import urlopen
urlopen("http://127.0.0.1:8000/health", timeout=2).read()
PY
}

bridge_healthy_stable() {
  bridge_healthy || return 1
  sleep 1
  bridge_healthy
}

wait_for_bridge() {
  local attempts="${1:-20}"
  local delay="${2:-1}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if bridge_healthy; then
      return 0
    fi
    sleep "${delay}"
  done
  return 1
}

tail_log_on_failure() {
  local label="$1"
  local log_file="$2"
  echo "${label} failed to start. Last log lines:"
  if [[ -s "${log_file}" ]]; then
    tail -n 80 "${log_file}" || true
  else
    echo "${log_file} is empty."
  fi
}

start_bridge() {
  local bridge_pid="data/run/apex_bridge.pid"
  local bridge_screen="${APEX_BRIDGE_SCREEN_SESSION:-apex_bridge}"
  if bridge_healthy_stable; then
    echo "APEX bridge already responding at http://127.0.0.1:8000"
    return 0
  fi
  if screen_session_alive "${bridge_screen}"; then
    echo "APEX bridge screen session already running: ${bridge_screen}"
    return 0
  fi
  if pid_alive "${bridge_pid}"; then
    echo "APEX bridge pid already running: $(cat "${bridge_pid}")"
    return 0
  fi

  if command -v screen >/dev/null 2>&1 && [[ "${APEX_LOCAL_START_METHOD:-screen}" == "screen" ]]; then
    screen -L -dmS "${bridge_screen}" "$(pwd -P)/scripts/run_local_bridge_service.sh"
    rm -f "${bridge_pid}"
    echo "Started APEX bridge screen session ${bridge_screen}"
  elif [[ "${APEX_LOCAL_START_STRATEGY:-false}" == "true" ]]; then
    if [[ "${LIVE_TRADING:-false}" != "true" ]]; then
      echo "LIVE_TRADING=true is required when APEX_LOCAL_START_STRATEGY=true."
      exit 1
    fi
    for key in TELEGRAM_CHAT_ID OPENAI_API_KEY; do
      if [[ -z "${!key:-}" ]]; then
        echo "Missing ${key}; refusing to start live strategy loop."
        exit 1
      fi
    done
    nohup env PYTHONPATH=. python3 -u scripts/start_bridge_prod.py >> logs/local_bridge.log 2>&1 &
  else
    nohup env PYTHONPATH=. python3 -u -m src.main --bridge-only >> logs/local_bridge.log 2>&1 &
    echo $! > "${bridge_pid}"
    echo "Started APEX bridge pid $(cat "${bridge_pid}")"
  fi

  if ! wait_for_bridge 20 1; then
    rm -f "${bridge_pid}"
    tail_log_on_failure "APEX bridge" "logs/local_bridge.log"
    exit 1
  fi
}

start_telegram_poll() {
  local poll_pid="data/run/apex_telegram_poll.pid"
  local poll_screen="${APEX_TELEGRAM_SCREEN_SESSION:-apex_telegram}"
  if telegram_poller_alive; then
    echo "APEX Telegram poller already running"
    return 0
  fi
  if screen_session_alive "${poll_screen}"; then
    echo "APEX Telegram poller screen session already running: ${poll_screen}"
    return 0
  fi
  if pid_alive "${poll_pid}"; then
    echo "APEX Telegram poller already running: $(cat "${poll_pid}")"
    return 0
  fi
  if command -v screen >/dev/null 2>&1 && [[ "${APEX_LOCAL_START_METHOD:-screen}" == "screen" ]]; then
    screen -L -dmS "${poll_screen}" "$(pwd -P)/scripts/run_local_telegram_service.sh"
    rm -f "${poll_pid}"
    echo "Started APEX Telegram poller screen session ${poll_screen}"
  else
    nohup env PYTHONPATH=. python3 -u scripts/apex_telegram_poll.py --claim-owner >> logs/telegram_poll.log 2>&1 &
    echo $! > "${poll_pid}"
    echo "Started APEX Telegram poller pid $(cat "${poll_pid}")"
  fi

  if [[ "${APEX_LOCAL_START_METHOD:-screen}" == "screen" ]] && command -v screen >/dev/null 2>&1; then
    if ! wait_for_screen_session "${poll_screen}" 30 1 && ! telegram_poller_alive; then
      tail_log_on_failure "APEX Telegram poller" "logs/telegram_poll.log"
      exit 1
    fi
  elif ! pid_alive "${poll_pid}"; then
    rm -f "${poll_pid}"
    tail_log_on_failure "APEX Telegram poller" "logs/telegram_poll.log"
    exit 1
  fi
}

start_bridge
start_telegram_poll

echo "Local Mac APEX services started."
echo "Bridge: http://127.0.0.1:8000/health"
echo "Logs: logs/local_bridge.log and logs/telegram_poll.log"
if [[ -z "${TELEGRAM_CHAT_ID:-}" ]]; then
  echo "Send /start to Nexus_vantage_trader_bot; the poller will claim TELEGRAM_CHAT_ID automatically."
fi
