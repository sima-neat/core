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
  if ! python3 - deps/manifest.json <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
data = json.loads(manifest_path.read_text(encoding="utf-8"))

value = data.get("internals")
if isinstance(value, str) and value.strip():
    raise SystemExit(0)
if isinstance(value, dict):
    policy = str(value.get("policy", "")).strip().lower()
    if policy == "snap":
        required = str(value.get("required-branch", "")).strip()
        if "required-branch" in value and not required:
            raise SystemExit(1)
        raise SystemExit(0)
    if value.get("required-branch"):
        raise SystemExit(1)
    spec = str(value.get("spec", "")).strip()
    branch = str(value.get("branch", value.get("ref", ""))).strip()
    if branch and (spec or value.get("spec", "") == ""):
        raise SystemExit(0)

raise SystemExit(1)
PY
  then
    echo "ERROR: deps/manifest.json must define internals as a non-empty string, {'policy':'snap'}, or {'branch':'...', 'spec':'...'}." >&2
    fail=1
  fi
  platform_version="$(sed -n 's/.*"platform-version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' deps/manifest.json | head -n1)"
  if [[ -z "${platform_version}" ]]; then
    echo "ERROR: deps/manifest.json must define a non-empty platform-version." >&2
    fail=1
  fi
  package_version="$(sed -n 's/.*"package-version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' deps/manifest.json | head -n1)"
  if [[ -z "${package_version}" ]]; then
    echo "ERROR: deps/manifest.json must define a non-empty package-version." >&2
    fail=1
  fi
  abi_version="$(sed -n 's/.*"abi-version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' deps/manifest.json | head -n1)"
  if [[ ! "${abi_version}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: deps/manifest.json abi-version must be a positive integer." >&2
    fail=1
  fi
  if ! python3 - deps/manifest.json <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "platform-package-version": "2.1.3~pre4593",
    "platform-package-contract": {
        "kernel": {
            "package": "linux-image-6.18.3-modalix",
            "package-version": "6.18.3-4593",
            "release": "6.18.3-modalix",
            "build-marker": "#4593",
        },
        "libcamera": {
            "package-version": "2.1.3+neat2",
            "platform-compat-version": "2.1.3~pre4593",
            "capability-name": "simaai-libcamera-dmabuf-abi",
            "capability-version": "1",
        },
        "memory": {
            "package-version": "2.1.1-0neat5",
            "platform-compat-version": "2.1.1~pre4593",
            "capability-name": "simaai-memory-dmabuf-export-abi",
            "capability-version": "1",
        },
    },
}
for key, value in expected.items():
    if data.get(key) != value:
        raise SystemExit(f"{path}: {key} must equal the reviewed B4593 contract")
PY
  then
    echo "ERROR: deps/manifest.json must contain the exact reviewed B4593 platform package contract." >&2
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
