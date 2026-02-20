#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "[release-policy] checking working tree cleanliness..."
status="$(git status --porcelain --untracked-files=all)"
if [[ -n "${status}" ]]; then
  echo "ERROR: Working tree is dirty. Commit or remove generated artifacts before release." >&2
  echo "${status}" >&2
  exit 1
fi

echo "[release-policy] clean branch check passed."
