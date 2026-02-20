#!/usr/bin/env bash
set -euo pipefail

fail=0

# 1) No internal headers should live under include/
if find include -type f -path "*/internal/*" | grep -q .; then
  echo "[error] Internal headers found under include/:" >&2
  find include -type f -path "*/internal/*" >&2
  fail=1
fi

# 2) Examples and tutorials must not include internal headers
internal_includes=$(grep -R --line-number -E '#include[[:space:]]*[<"][^">]*internal/' examples tutorials 2>/dev/null || true)
if [ -n "${internal_includes}" ]; then
  echo "[error] Internal headers included from examples/tutorials:" >&2
  echo "${internal_includes}" >&2
  fail=1
fi

if [ "${fail}" -ne 0 ]; then
  exit 1
fi

echo "[ok] Internal header hygiene checks passed."
