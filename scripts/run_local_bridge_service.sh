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
export PYTHONUNBUFFERED=1

mkdir -p data/run logs

if [[ "${APEX_SERVICE_LOG_REDIRECT:-true}" == "true" && -z "${APEX_BRIDGE_LOG_REDIRECTED:-}" ]]; then
  export APEX_BRIDGE_LOG_REDIRECTED=1
  exec "$0" >> logs/local_bridge.log 2>&1
fi

if [[ "${APEX_LOCAL_START_STRATEGY:-false}" == "true" ]]; then
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
  exec env PYTHONPATH=. python3 -u scripts/start_bridge_prod.py
fi

exec env PYTHONPATH=. python3 -u -m src.main --bridge-only
