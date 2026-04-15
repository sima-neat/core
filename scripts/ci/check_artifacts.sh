#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MAX_FILE_SIZE_BYTES="${MAX_FILE_SIZE_BYTES:-5242880}"

allow_large_file() {
  local path="$1"
  case "${path}" in
    tests/assets/*) return 0 ;;
    tests/images/*) return 0 ;;
    docs/images/*) return 0 ;;
    test.jpg) return 0 ;;
  esac
  return 1
}

file_size_bytes() {
  local path="$1"
  if stat --version >/dev/null 2>&1; then
    stat -c%s "${path}"
  else
    stat -f%z "${path}"
  fi
}

fail=0

echo "[repo-hygiene] checking deps manifest..."
if [[ ! -f deps/manifest.json ]]; then
  echo "ERROR: deps/manifest.json is required." >&2
  fail=1
else
  internals_ref="$(sed -n 's/.*"internals"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' deps/manifest.json | head -n1)"
  if [[ -z "${internals_ref}" ]]; then
    echo "ERROR: deps/manifest.json must define a non-empty internals." >&2
    fail=1
  fi
fi

echo "[repo-hygiene] checking tracked binary artifacts in source paths..."
while IFS= read -r path; do
  [[ -z "${path}" ]] && continue

  if [[ "${path}" =~ ^(src|include|examples|tutorials|docs|website|scripts)/ ]] && \
     [[ "${path}" =~ \.(so|a|o|out|dylib|dll|exe|bin|bak)$ ]]; then
    echo "ERROR: Forbidden binary artifact tracked in source path: ${path}" >&2
    fail=1
  fi

  if [[ "${path}" == "core" || "${path}" == "a.out" ]]; then
    echo "ERROR: Forbidden runtime artifact tracked at repo root: ${path}" >&2
    fail=1
  fi

  if [[ ! -f "${path}" ]]; then
    continue
  fi

  size_bytes="$(file_size_bytes "${path}")"
  if (( size_bytes > MAX_FILE_SIZE_BYTES )); then
    if ! allow_large_file "${path}"; then
      echo "ERROR: Large tracked file exceeds ${MAX_FILE_SIZE_BYTES} bytes: ${path} (${size_bytes})" >&2
      fail=1
    fi
  fi
done < <(git ls-files)

if git ls-files | grep -q '^third_party/gst-plugins/'; then
  echo "ERROR: tracked binaries under third_party/gst-plugins are not allowed." >&2
  fail=1
fi

if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

echo "[repo-hygiene] artifact checks passed."
