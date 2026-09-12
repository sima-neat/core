#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

expected="$({
  python3 - "${WORKSPACE}/deps/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as manifest_file:
    version = str(json.load(manifest_file).get("platform-version", "")).strip()
if not version:
    raise SystemExit("deps/manifest.json does not define platform-version")
print(version.split("+", 1)[0])
PY
})"

actual="$(ssh_card "awk -F= '\
  \$1 ~ /^[[:space:]]*DISTRO_VERSION[[:space:]]*\$/ { \
    value=\$2; \
    sub(/^[[:space:]]+/, \"\", value); \
    sub(/[[:space:]]+\$/, \"\", value); \
    print value; \
    exit \
  }' /etc/buildinfo")"

if [[ -z "${actual}" ]]; then
  echo "ERROR: PCIe card /etc/buildinfo does not define DISTRO_VERSION." >&2
  exit 1
fi

if [[ "${actual}" == "${expected}" ]]; then
  echo "PCIe card platform is compatible: ${actual}"
else
  write_state_var PCIE_CARD_PLATFORM_EXPECTED "${expected}"
  write_state_var PCIE_CARD_PLATFORM_ACTUAL "${actual}"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    printf 'compatible=false\n' >>"${GITHUB_OUTPUT}"
  fi
  echo "ERROR: PCIe card platform ${actual} is incompatible with package platform ${expected}." >&2
  exit 1
fi

write_state_var PCIE_CARD_PLATFORM_EXPECTED "${expected}"
write_state_var PCIE_CARD_PLATFORM_ACTUAL "${actual}"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'compatible=true\n' >>"${GITHUB_OUTPUT}"
fi
