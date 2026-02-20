#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

fail=0

echo "[repo-hygiene] checking for unresolved merge markers..."
conflicts="$(git grep -nE '^(<<<<<<<|=======|>>>>>>>)( .*)?$' -- . \
  ':(exclude)docs/doxygen/**' \
  ':(exclude)website/build/**' \
  ':(exclude)website/.docusaurus/**' || true)"

if [[ -n "${conflicts}" ]]; then
  echo "ERROR: Merge conflict markers found:" >&2
  echo "${conflicts}" >&2
  fail=1
fi

echo "[repo-hygiene] checking canonical naming in user-facing docs..."
legacy_matches="$(git grep -nE '\<(PipelineSession|PipelineRun|NeatModel|NeatTensor|InputAppSrc|OutputAppSink)\>' -- README.md docs \
  ':(exclude)docs/doxygen/**' \
  ':(exclude)docs/reference/cppapi/**' \
  ':(exclude)docs/how-to/migration_legacy_names.md' \
  ':(exclude)docs/contribute/naming.md' || true)"

legacy_matches="$(printf '%s\n' "${legacy_matches}" | grep -vE 'github\.com/.*/PipelineSession' || true)"

if [[ -n "${legacy_matches}" ]]; then
  echo "ERROR: Legacy names found in public docs:" >&2
  echo "${legacy_matches}" >&2
  echo "Allowed legacy references must be restricted to migration docs only." >&2
  fail=1
fi

if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

echo "[repo-hygiene] naming and conflict checks passed."
