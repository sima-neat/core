#!/usr/bin/env bash
# Run inside the verified SDK container before build.sh. Only lock contention
# is retryable: 25 attempts, with at most 120 seconds of sleep between attempts.
# Package configuration itself is not timed out or killed.
set -euo pipefail
export LC_ALL=C

log=$(mktemp)
trap 'rm -f "$log"' EXIT
max_attempts=25
retry_seconds=5
lock_error='^dpkg: error: (dpkg frontend lock|dpkg database lock|dpkg database area) (was locked|is locked) by another process'

report_lock_owner() {
  local owner parent
  owner=$(sed -nE 's/.*locked by another process with pid ([0-9]+).*/\1/p' "$log" | head -n 1)
  if [[ -n "$owner" ]]; then
    echo "SDK dpkg lock owner PID $owner (snapshot; process may have exited):" >&2
    ps -p "$owner" -o pid,ppid,etime,args >&2 || true
    parent=$(ps -p "$owner" -o ppid= 2>/dev/null | tr -d '[:space:]') || parent=''
    if [[ "$parent" =~ ^[0-9]+$ ]]; then
      echo 'SDK dpkg lock owner parent:' >&2
      ps -p "$parent" -o pid,ppid,etime,args >&2 || true
    fi
  else
    echo 'No owner PID in dpkg error; package-manager process snapshot:' >&2
    # Retain parent and elapsed-time columns for diagnosing startup races.
    # shellcheck disable=SC2009
    ps -eo pid,ppid,etime,args 2>/dev/null | grep -E '[a]pt|[d]pkg|[u]nattended' >&2 || true
  fi
}

for ((attempt=1; attempt<=max_attempts; attempt++)); do
  if dpkg --configure -a >"$log" 2>&1; then
    cat "$log"
    exit 0
  else
    status=$?
  fi
  cat "$log" >&2
  # Do not retry a failure that also reports an unrelated dpkg error.
  if ((status != 2)) || ! grep -Eq "$lock_error" "$log" ||
      { grep '^dpkg: error:' "$log" | grep -Evq "$lock_error"; }; then
    exit "$status"
  fi
  echo "SDK dpkg lock contention (attempt $attempt/$max_attempts)." >&2
  report_lock_owner
  if ((attempt == max_attempts)); then
    echo 'SDK dpkg lock retry budget exhausted after 25 attempts (120 seconds of retry sleep).' >&2
    exit "$status"
  fi
  sleep "$retry_seconds"
done
