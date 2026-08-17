#!/usr/bin/env bash

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TEMP_ROOT}"' EXIT

mkdir -p "${TEMP_ROOT}/bin"

cat >"${TEMP_ROOT}/bin/sima-cli" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exit 0
EOF

cat >"${TEMP_ROOT}/bin/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == "-c" ]]
[[ "${2:-}" == *"login_external(loginDocker=False)"* ]]
exit "${FAKE_SIMA_CLI_LOGIN_STATUS:-0}"
EOF

cat >"${TEMP_ROOT}/bin/timeout" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
shift
exec "$@"
EOF

chmod +x "${TEMP_ROOT}/bin/"*

output="$({
  PATH="${TEMP_ROOT}/bin:${PATH}" \
    "${REPO_ROOT}/scripts/ci/check_sima_cli_session.sh"
} 2>&1)"
[[ "${output}" == *"session is authenticated"* ]]

set +e
output="$({
  PATH="${TEMP_ROOT}/bin:${PATH}" \
    FAKE_SIMA_CLI_LOGIN_STATUS=1 \
    "${REPO_ROOT}/scripts/ci/check_sima_cli_session.sh"
} 2>&1)"
status=$?
set -e

[[ "${status}" -eq 1 ]]
[[ "${output}" == *"authentication preflight failed"* ]]

echo "sima-cli session preflight tests passed"
