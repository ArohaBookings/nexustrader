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
    if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    key="${key%"${key##*[![:space:]]}"}"
    key="${key#"${key%%[![:space:]]*}"}"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -z "${!key+x}" || -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${env_file}"
}

for env_file in "config/secrets.env" "secrets.env"; do
  load_env_file "${env_file}"
done

for key in TELEGRAM_BOT_TOKEN TELEGRAM_WEBHOOK_SECRET; do
  if [[ -z "${!key:-}" ]]; then
    echo "Missing ${key}; refusing to start Telegram poller."
    exit 1
  fi
done

export PYTHONUNBUFFERED=1
mkdir -p data/run logs

if [[ "${APEX_SERVICE_LOG_REDIRECT:-true}" == "true" && -z "${APEX_TELEGRAM_LOG_REDIRECTED:-}" ]]; then
  export APEX_TELEGRAM_LOG_REDIRECTED=1
  exec "$0" >> logs/telegram_poll.log 2>&1
fi

exec env PYTHONPATH=. python3 -u scripts/apex_telegram_poll.py --claim-owner
