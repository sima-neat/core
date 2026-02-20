#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: run_clean_env <command...>" >&2
  exit 2
fi

# Run without sourcing /etc/profile.d (avoids resize-serial noise).
exec /bin/bash --noprofile --norc -c "$*"
