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
    case "${key}" in
      INGEST_API_KEY|NEXUS_INGEST_URL|BRIDGE_API_BASE_URL|BRIDGE_TOKEN|BRIDGE_DASHBOARD_PASSWORD|APEX_DASHBOARD_PASSWORD|APP_ADMIN_PASSWORD|COLLECTOR_INTERVAL_MS)
        ;;
      *)
        continue
        ;;
    esac
    if [[ -z "${!key+x}" || -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${env_file}"
}

for env_file in \
  "config/secrets.env" \
  "secrets.env" \
  "nexus-trader/.env.production.local" \
  "nexus-trader/.env.local"; do
  load_env_file "${env_file}"
done

export BRIDGE_API_BASE_URL="${BRIDGE_API_BASE_URL:-http://127.0.0.1:8000}"
export NEXUS_INGEST_URL="${NEXUS_INGEST_URL:-https://nexustrader-flax.vercel.app/api/ingest}"
export COLLECTOR_INTERVAL_MS="${COLLECTOR_INTERVAL_MS:-15000}"

mkdir -p logs data/run
exec >> logs/nexus_collector.log 2>&1

if [[ -z "${INGEST_API_KEY:-}" ]]; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) nexus_collector skipped: INGEST_API_KEY missing"
  exit 0
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) nexus_collector starting: ${BRIDGE_API_BASE_URL} -> ${NEXUS_INGEST_URL}"
unset VERCEL VERCEL_ENV VERCEL_GIT_COMMIT_AUTHOR_LOGIN VERCEL_GIT_COMMIT_AUTHOR_NAME
unset VERCEL_GIT_COMMIT_MESSAGE VERCEL_GIT_COMMIT_REF VERCEL_GIT_COMMIT_SHA VERCEL_GIT_PREVIOUS_SHA
unset VERCEL_GIT_PROVIDER VERCEL_GIT_PULL_REQUEST_ID VERCEL_GIT_REPO_ID VERCEL_GIT_REPO_OWNER VERCEL_GIT_REPO_SLUG
unset VERCEL_OIDC_TOKEN VERCEL_TARGET_ENV VERCEL_URL NX_DAEMON TURBO_CACHE TURBO_REMOTE_ONLY TURBO_RUN_SUMMARY TURBO_DOWNLOAD_LOCAL_ENABLED
cd nexus-trader
exec npm run collector
