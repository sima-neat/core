#!/usr/bin/env bash
set -euo pipefail

MODE="changed"
BASE_REF="${HYGIENE_BASE_REF:-}"
FORMAT_VERBOSE=0

usage() {
  cat <<USAGE
Usage: scripts/ci/run_code_hygiene.sh [--changed-only|--all] [--base-ref <ref>] [--format-verbose]
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
    --format-verbose)
      FORMAT_VERBOSE=1
      shift
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
if [[ "${MODE}" == "all" ]]; then
  args+=(--all)
else
  args+=(--changed-only)
fi
if [[ -n "${BASE_REF}" ]]; then
  args+=(--base-ref "${BASE_REF}")
fi

format_args=("${args[@]}")
if [[ "${FORMAT_VERBOSE}" -eq 1 ]]; then
  format_args+=(--verbose)
fi

echo "[code-hygiene] checking internal header boundaries"
./scripts/check_internal_headers.sh

echo "[code-hygiene] checking C/C++ formatting"
./scripts/check_format.sh "${format_args[@]}"

echo "[code-hygiene] checking CMake style"
./scripts/check_cmake_format.sh "${args[@]}"

echo "[code-hygiene] checking duplicate includes"
./scripts/check_duplicate_includes.sh "${args[@]}"

echo "[code-hygiene] checking clang-tidy"
SIMANEAT_STRICT_WARNINGS="${SIMANEAT_STRICT_WARNINGS:-ON}" ./scripts/run_cpp_tidy.sh "${args[@]}"

echo "[code-hygiene] OK"
