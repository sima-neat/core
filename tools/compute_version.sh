#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

version=""

if git describe --tags --exact-match >/dev/null 2>&1; then
  version="$(git describe --tags --exact-match 2>/dev/null || true)"
else
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  hash="$(git rev-parse --short HEAD 2>/dev/null || true)"
  if [[ -n "${branch}" && -n "${hash}" ]]; then
    version="${branch}-${hash}"
  fi
fi

if [[ -z "${version}" ]]; then
  version="0.0.0"
fi

# Normalize: strip leading 'v' and replace unsupported chars.
if [[ "${version}" =~ ^v[0-9] ]]; then
  version="${version#v}"
fi

version="${version//[^A-Za-z0-9.+~_-]/-}"

if [[ ! "${version}" =~ ^[0-9] ]]; then
  version="0.0.0+${version}"
fi

echo "${version}"
