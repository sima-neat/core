#!/usr/bin/env bash
set -euo pipefail

MODE="changed"
BASE_REF="${FORMAT_BASE_REF:-}"

usage() {
  cat <<USAGE
Usage: scripts/check_duplicate_includes.sh [--changed-only|--all] [--base-ref <ref>]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --changed-only)
      MODE="changed"
      shift
      ;;
    --all)
      MODE="all"
      shift
      ;;
    --base-ref)
      BASE_REF="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

args=()
if [[ "$MODE" == "all" ]]; then
  args+=(--all)
else
  args+=(--changed-only)
fi
if [[ -n "$BASE_REF" ]]; then
  args+=(--base-ref "$BASE_REF")
fi

python3 scripts/check_duplicate_includes.py "${args[@]}"
