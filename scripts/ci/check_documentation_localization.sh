#!/usr/bin/env bash
# Validate committed documentation and tutorial translations before publication.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

npm --prefix website run check:i18n-complete
python3 tools/generate_tutorial_docs.py --repo-root .
npm --prefix website run check:i18n-navigation
