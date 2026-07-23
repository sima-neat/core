#!/usr/bin/env bash

set -euo pipefail

readonly AUTH_CHECK_TIMEOUT_SECONDS="${SIMA_CLI_AUTH_CHECK_TIMEOUT_SECONDS:-20}"

sima_cli_bin="$(command -v sima-cli || true)"
if [[ -z "${sima_cli_bin}" ]]; then
  echo "::error::sima-cli is not available on PATH; cannot validate Developer Portal authentication." >&2
  exit 1
fi

sima_cli_python="$(dirname "${sima_cli_bin}")/python"
if [[ ! -x "${sima_cli_python}" ]]; then
  echo "::error::Unable to locate the Python interpreter used by sima-cli: ${sima_cli_python}" >&2
  exit 1
fi

echo "Validating the saved sima-cli Developer Portal session (timeout: ${AUTH_CHECK_TIMEOUT_SECONDS}s)."

set +e
timeout "${AUTH_CHECK_TIMEOUT_SECONDS}s" \
  "${sima_cli_python}" -c \
  'from sima_cli.auth.devportal import validate_session; _, valid = validate_session(); raise SystemExit(0 if valid else 1)'
status=$?
set -e

if [[ "${status}" -eq 0 ]]; then
  echo "sima-cli Developer Portal session is authenticated."
  exit 0
fi

if [[ "${status}" -eq 124 ]]; then
  reason="the session check timed out after ${AUTH_CHECK_TIMEOUT_SECONDS}s"
else
  reason="the saved session is missing, expired, or does not have Developer Portal access"
fi

cat >&2 <<EOF
::error::sima-cli authentication preflight failed: ${reason}.
Authenticate as the GitHub Actions runner service account on $(hostname) before rerunning CI:
  sudo -u "$(id -un)" -H sima-cli login
The E2E tests were not started, avoiding model-download retries and test timeouts.
EOF
exit 1
