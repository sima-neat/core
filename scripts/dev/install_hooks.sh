#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

GIT_HOOKS_DIR="$(git rev-parse --git-path hooks)"
mkdir -p "${GIT_HOOKS_DIR}"

for hook_name in pre-commit pre-push; do
  HOOK_SRC="${ROOT_DIR}/.githooks/${hook_name}"
  if [[ ! -f "${HOOK_SRC}" ]]; then
    echo "ERROR: Missing hook source: ${HOOK_SRC}" >&2
    exit 1
  fi
  install -m 0755 "${HOOK_SRC}" "${GIT_HOOKS_DIR}/${hook_name}"
  echo "Installed ${hook_name} hook at ${GIT_HOOKS_DIR}/${hook_name}"
done
