#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${root_dir}/build-tidy"
MODE="${TIDY_MODE:-changed}"
BASE_REF="${TIDY_BASE_REF:-}"

usage() {
  cat <<USAGE
Usage: scripts/run_cpp_tidy.sh [--changed-only|--all] [--base-ref <ref>]
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
require_cmd git

configure_tidy_build() {
  local cmake_prefix_path="${CMAKE_PREFIX_PATH:-}"
  local candidate

  for candidate in \
    "${SYSROOT:-}/usr" \
    "${SDKTARGETSYSROOT:-}/usr" \
    "/opt/toolchain/aarch64/modalix/usr"; do
    [[ -n "$candidate" ]] || continue
    if [[ -f "${candidate}/lib/aarch64-linux-gnu/cmake/SimaLMM/SimaLMMConfig.cmake" ]]; then
      if [[ -n "$cmake_prefix_path" ]]; then
        cmake_prefix_path="${cmake_prefix_path}:${candidate}"
      else
        cmake_prefix_path="${candidate}"
      fi
    fi
  done

  # Keep clang-tidy analysis on the host toolchain even when SDK/cross env
  # variables are present in CI.
  env \
    -u CC \
    -u CXX \
    -u CPPFLAGS \
    -u CFLAGS \
    -u CXXFLAGS \
    -u LDFLAGS \
    -u SYSROOT \
    -u SDKTARGETSYSROOT \
    -u PKG_CONFIG_SYSROOT_DIR \
    -u PKG_CONFIG_PATH \
    -u PKG_CONFIG_LIBDIR \
    -u CONFIGURE_FLAGS \
    -u OECORE_TARGET_ARCH \
    -u OECORE_TARGET_SYSROOT \
    -u OECORE_NATIVE_SYSROOT \
    CMAKE_PREFIX_PATH="${cmake_prefix_path}" \
    cmake -S "$root_dir" -B "$build_dir" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DSIMANEAT_CPP_TIDY=ON \
      -DSIMANEAT_STRICT_WARNINGS=ON \
      -DSIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS=OFF
}

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

resolve_base_ref() {
  local base="$BASE_REF"
  if [[ -n "$base" ]]; then
    echo "$base"
    return 0
  fi

  if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
    local remote_ref="origin/${GITHUB_BASE_REF}"
    if ! git rev-parse --verify --quiet "$remote_ref" >/dev/null; then
      git fetch --no-tags --depth=1 origin "${GITHUB_BASE_REF}:${remote_ref}" >/dev/null 2>&1 || true
    fi
    if git rev-parse --verify --quiet "$remote_ref" >/dev/null; then
      echo "$remote_ref"
      return 0
    fi
  fi

  if [[ -n "${CI:-}" && "${CI:-}" != "false" && "${CI:-}" != "0" ]]; then
    if git rev-parse --verify --quiet HEAD~1 >/dev/null; then
      echo "HEAD~1"
      return 0
    fi
  fi

  echo ""
}

collect_changed() {
  local base
  base="$(resolve_base_ref)"
  if [[ -n "$base" ]]; then
    git diff --name-only --diff-filter=ACMRTUXB "$base"...HEAD
  else
    git diff --name-only --diff-filter=ACMRTUXB --cached
  fi
}

is_cpp_source() {
  case "$1" in
    *.c|*.cc|*.cpp|*.cxx) return 0 ;;
    *) return 1 ;;
  esac
}

is_tidy_scope_path() {
  case "$1" in
    include/*|src/*|tutorials/*|examples/*|tests/*) return 0 ;;
    *) return 1 ;;
  esac
}

configure_tidy_build

if [[ "$MODE" == "all" ]]; then
  mapfile -t all_files < <(rg --no-messages --no-config --files \
    -g '*.c' -g '*.cc' -g '*.cpp' -g '*.cxx' \
    "$root_dir/include" "$root_dir/src" "$root_dir/tutorials" "$root_dir/examples" "$root_dir/tests" || true)
else
  mapfile -t candidates < <(collect_changed || true)
  all_files=()
  for f in "${candidates[@]}"; do
    [[ -f "$f" ]] || continue
    is_tidy_scope_path "$f" || continue
    is_cpp_source "$f" || continue
    all_files+=("$f")
  done
fi

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

echo "clang-tidy mode: ${MODE}"
echo "clang-tidy file count: ${#files[@]}"
if [ ${#files[@]} -eq 0 ]; then
  echo "No non-GStreamer sources found for clang-tidy (${MODE} mode)."
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
