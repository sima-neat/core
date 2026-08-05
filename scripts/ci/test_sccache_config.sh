#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
temp_root="$(mktemp -d)"
trap 'rm -rf "${temp_root}"' EXIT

mkdir -p "${temp_root}/bin" "${temp_root}/home"
cat >"${temp_root}/bin/sccache" <<'SCCACHE'
#!/usr/bin/env bash
case "${1:-}" in
  --version) echo "sccache 0.test" ;;
  --zero-stats)
    if [[ "${FAKE_SCCACHE_ZERO_STATS_FAILURE:-OFF}" == "ON" ]]; then
      echo "simulated sccache startup failure" >&2
      exit 1
    fi
    ;;
  --show-stats)
    if [[ "${FAKE_SCCACHE_SHOW_STATS_FAILURE:-OFF}" == "ON" ]]; then
      echo "simulated sccache statistics failure" >&2
      exit 1
    fi
    ;;
  *) echo "unexpected fake sccache argument: ${1:-}" >&2; exit 1 ;;
esac
SCCACHE
chmod +x "${temp_root}/bin/sccache"

(
  export HOME="${temp_root}/home"
  export PATH="${temp_root}/bin:${PATH}"
  export SIMANEAT_SCCACHE=off
  # shellcheck source=scripts/configure_sccache.sh
  source "${repo_root}/scripts/configure_sccache.sh"
  simaneat_configure_sccache "${repo_root}"
  [[ "${SIMANEAT_SCCACHE_ACTIVE}" == "OFF" ]]
)

if (
  export HOME="${temp_root}/home"
  export PATH="${temp_root}/bin:${PATH}"
  export SIMANEAT_SCCACHE=on
  export SIMANEAT_SCCACHE_ZERO_STATS=ON
  export FAKE_SCCACHE_ZERO_STATS_FAILURE=ON
  # shellcheck source=scripts/configure_sccache.sh
  source "${repo_root}/scripts/configure_sccache.sh"
  simaneat_configure_sccache "${repo_root}"
); then
  echo "Expected sccache startup failure to remain fatal by default." >&2
  exit 1
fi

(
  export HOME="${temp_root}/home"
  export PATH="${temp_root}/bin:${PATH}"
  export SIMANEAT_SCCACHE=on
  export SIMANEAT_SCCACHE_ZERO_STATS=ON
  export SIMANEAT_SCCACHE_STARTUP_FAIL_OPEN=ON
  export FAKE_SCCACHE_ZERO_STATS_FAILURE=ON
  # shellcheck source=scripts/configure_sccache.sh
  source "${repo_root}/scripts/configure_sccache.sh"
  simaneat_configure_sccache "${repo_root}"
  [[ "${SIMANEAT_SCCACHE_ACTIVE}" == "OFF" ]]
)

(
  export HOME="${temp_root}/home"
  export PATH="${temp_root}/bin:${PATH}"
  export SIMANEAT_SCCACHE=on
  export SIMANEAT_SCCACHE_ZERO_STATS=ON
  # shellcheck source=scripts/configure_sccache.sh
  source "${repo_root}/scripts/configure_sccache.sh"
  simaneat_configure_sccache "${repo_root}"
  [[ "${SIMANEAT_SCCACHE_ACTIVE}" == "ON" ]]
  [[ "${SIMANEAT_SCCACHE_BIN}" == "${temp_root}/bin/sccache" ]]
  [[ "${SCCACHE_DIR}" == "${temp_root}/home/.cache/sima-neat/sccache" ]]
  [[ "${SCCACHE_LOCAL_RW_MODE}" == "READ_WRITE" ]]
)

(
  export HOME="${temp_root}/home"
  export PATH="${temp_root}/bin:${PATH}"
  export SIMANEAT_SCCACHE=on
  export SCCACHE_BUCKET=test-bucket
  export SCCACHE_REGION=us-west-2
  export SCCACHE_S3_KEY_PREFIX=core/sccache-v1/arm64/sdk-develop/standard
  export SCCACHE_S3_RW_MODE=READ_ONLY
  # shellcheck source=scripts/configure_sccache.sh
  source "${repo_root}/scripts/configure_sccache.sh"
  simaneat_configure_sccache "${repo_root}"
  [[ "${SCCACHE_MULTILEVEL_CHAIN}" == "disk,s3" ]]
  [[ "${SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY}" == "l0" ]]
  [[ "${SCCACHE_S3_USE_SSL}" == "true" ]]
)

(
  export HOME="${temp_root}/home"
  export PATH="${temp_root}/bin:${PATH}"
  export SIMANEAT_SCCACHE_ACTIVE=ON
  export SIMANEAT_SCCACHE_BIN="${temp_root}/bin/sccache"
  export FAKE_SCCACHE_SHOW_STATS_FAILURE=ON
  # shellcheck source=scripts/configure_sccache.sh
  source "${repo_root}/scripts/configure_sccache.sh"
  simaneat_show_sccache_stats
)

if (
  export HOME="${temp_root}/home"
  export PATH="${temp_root}/bin:${PATH}"
  export SIMANEAT_SCCACHE=invalid
  # shellcheck source=scripts/configure_sccache.sh
  source "${repo_root}/scripts/configure_sccache.sh"
  simaneat_configure_sccache "${repo_root}"
); then
  echo "Expected invalid SIMANEAT_SCCACHE mode to fail." >&2
  exit 1
fi

echo "sccache configuration tests passed"
