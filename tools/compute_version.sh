#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

platform_version="$(python3 - <<'PY'
import json
from pathlib import Path

manifest_path = Path("deps/manifest.json")
data = json.loads(manifest_path.read_text(encoding="utf-8"))
version = str(data.get("platform-version", "")).strip()
if not version:
    raise SystemExit(f"Missing or empty 'platform-version' in {manifest_path}")
print(version)
PY
)"

version="${platform_version}"

if git describe --tags --exact-match >/dev/null 2>&1; then
  version="$(git describe --tags --exact-match 2>/dev/null || true)"
else
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  hash="$(git rev-parse --short HEAD 2>/dev/null || true)"
  if [[ -n "${branch}" && -n "${hash}" ]]; then
    branch="$(
      printf '%s' "${branch}" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's#[^a-z0-9.]+#.#g; s#^\.+##; s#\.+$##'
    )"
    if [[ -z "${branch}" ]]; then
      branch="branch"
    fi
    version="${platform_version}+${branch}.${hash}"
  fi
fi

# Normalize: strip leading 'v' and replace unsupported chars.
if [[ "${version}" =~ ^v[0-9] ]]; then
  version="${version#v}"
fi

version="$(printf '%s' "${version}" | sed -E 's#[^A-Za-z0-9.+~-]+#-#g')"

echo "${version}"
