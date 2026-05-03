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

if [[ "${LIVE_TRADING:-false}" != "true" ]]; then
  echo "LIVE_TRADING=true is required before live mode can start."
  exit 1
fi

required_live_env=(
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
  TELEGRAM_WEBHOOK_SECRET
  OPENAI_API_KEY
)
missing_live_env=()
for key in "${required_live_env[@]}"; do
  if [[ -z "${!key:-}" ]]; then
    missing_live_env+=("${key}")
  fi
done
if (( ${#missing_live_env[@]} > 0 )); then
  printf 'Missing required live env keys: %s\n' "${missing_live_env[*]}"
  echo "Populate config/secrets.env before live startup. Secret values were not printed."
  exit 1
fi

export APEX_MODE=LIVE
PYTHONPATH=. python3 scripts/start_bridge_prod.py
