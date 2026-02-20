#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${root_dir}/build-tidy"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd cmake
require_cmd clang-tidy
require_cmd run-clang-tidy
require_cmd rg

detect_jobs() {
  if [ -n "${TIDY_JOBS:-}" ]; then
    printf '%s\n' "${TIDY_JOBS}"
    return
  fi

  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi

  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
    return
  fi

  printf '1\n'
}

cmake -S "$root_dir" -B "$build_dir" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DSIMANEAT_CPP_TIDY=ON \
  -DSIMANEAT_STRICT_WARNINGS=ON

mapfile -t all_files < <(rg --no-messages --no-config --files \
  -g '*.c' -g '*.cc' -g '*.cpp' -g '*.cxx' \
  "$root_dir/include" "$root_dir/src" "$root_dir/tutorials" "$root_dir/examples" "$root_dir/tests" || true)

declare -A gst_excluded=()
if [ ${#all_files[@]} -gt 0 ]; then
  while IFS= read -r f; do
    gst_excluded["$f"]=1
  done < <(rg --no-messages --no-config -l '#include <gst|#include "gst/' "${all_files[@]}" || true)
fi

files=()
for f in "${all_files[@]}"; do
  if [ -z "${gst_excluded[$f]+x}" ]; then
    files+=("$f")
  fi
done

echo "clang-tidy file count: ${#files[@]}"
if [ ${#files[@]} -eq 0 ]; then
  echo "No non-GStreamer sources found for clang-tidy."
  exit 0
fi

jobs="$(detect_jobs)"
echo "clang-tidy parallel jobs: ${jobs}"
quiet="${TIDY_QUIET:-1}"
echo "clang-tidy quiet mode: ${quiet}"

printf '%s\n' "${files[@]}" | sed -n '1,20p'
tidy_args=(-p "$build_dir" -j "$jobs")
if [ "${quiet}" = "1" ]; then
  tidy_args+=(-quiet)
fi
run-clang-tidy "${tidy_args[@]}" "${files[@]}"
