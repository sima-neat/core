#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

hits="$(mktemp)"
unallowed="$(mktemp)"
trap 'rm -f "$hits" "$unallowed"' EXIT

# Hard gate: removed public names must only appear in migration stubs,
# negative tests, migration docs, or explicit ABI allowlist entries.
git ls-files -z \
  README.md CONTRIBUTING.md include src tests python docs tutorials distribution tools \
| xargs -0 rg -n --no-messages \
  '\b(Session|SessionOptions|SessionReport|SessionError|ModelSessionOptions|GraphSession|PullSession)\b|pyneat\.Session|model\.session\(' \
  -g '!docs/doxygen/out/**' \
  -g '!docs/reference/cppapi/**' \
  -g '!docs/reference/pythonapi/**' \
  -g '!docs/tutorials/tutorial_*.md' \
  -g '!docs/tutorials/tutorial_*.mdx' \
  -g '!docs/tutorials/index.md' \
  -g '!**/__pycache__/**' \
  -g '!*.pyc' \
  > "$hits" || true

# Keep this allowlist intentionally small and visible.
grep -Ev \
  '(^include/graph/GraphSession.h|python/pyneat/__init__.py|python/tests/test_api_surface.py|docs/contribute/naming.md|release|migration|SimaPluginStaticManifest|SimaPreparedRuntimeAbi|PreparedRuntimeAbi|session_id)' \
  "$hits" > "$unallowed" || true

if [[ -s "$unallowed" ]]; then
  echo "Unallowed stale public Session vocabulary remains:" >&2
  cat "$unallowed" >&2
  exit 1
fi

# Soft lowercase audit. This intentionally does not fail because ABI fields,
# perf scenario ids, migration docs, and private sentinels may still contain
# lowercase session.
git ls-files -z include src tests python docs tutorials distribution tools \
| xargs -0 rg -n --no-messages 'session' \
  -g '!docs/doxygen/out/**' \
  -g '!docs/reference/cppapi/**' \
  -g '!docs/reference/pythonapi/**' \
  -g '!docs/tutorials/tutorial_*.md' \
  -g '!docs/tutorials/tutorial_*.mdx' \
  -g '!docs/tutorials/index.md' \
  -g '!**/__pycache__/**' \
  -g '!*.pyc' \
  || true
