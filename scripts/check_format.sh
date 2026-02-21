#!/usr/bin/env bash
set -euo pipefail

MODE="changed"
BASE_REF="${FORMAT_BASE_REF:-}"
VERBOSE=0
PATH_FILTER="${FORMAT_PATH_FILTER:-}"

usage() {
  cat <<USAGE
Usage: scripts/check_format.sh [--changed-only|--all] [--base-ref <ref>]
       [--path <folder>] [--verbose]
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
    --verbose)
      VERBOSE=1
      shift
      ;;
    --path)
      PATH_FILTER="${2:-}"
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

normalize_path_filter() {
  local p="$1"
  local root
  root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

  # Convert absolute paths under the repo to repo-relative paths.
  if [[ "$p" == "$root" ]]; then
    echo ""
    return 0
  fi
  if [[ "$p" == "$root/"* ]]; then
    p="${p#"$root"/}"
  fi

  p="${p#./}"
  p="${p%/}"
  if [[ "$p" == "." ]]; then
    p=""
  fi
  echo "$p"
}

if [[ -n "$PATH_FILTER" ]]; then
  local_path_filter_input="$PATH_FILTER"
  PATH_FILTER="$(normalize_path_filter "$PATH_FILTER")"
  if [[ -z "$PATH_FILTER" && "$local_path_filter_input" != "." ]]; then
    root_path="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    if [[ "$local_path_filter_input" != "$root_path" && "$local_path_filter_input" != "$root_path/" ]]; then
      echo "ERROR: --path requires a non-empty folder path." >&2
      exit 1
    fi
  fi
fi

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

  # Local hooks should check staged changes by default. Only use commit-range
  # fallback automatically in CI, where no staged index is present.
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

is_cpp_file() {
  case "$1" in
    *.c|*.cc|*.cpp|*.cxx|*.h|*.hpp) return 0 ;;
    *) return 1 ;;
  esac
}

mapfile -t candidates < <(
  if [[ "$MODE" == "all" ]]; then
    git ls-files
  else
    collect_changed
  fi
)

files=()
for f in "${candidates[@]}"; do
  [[ -f "$f" ]] || continue
  if [[ -n "$PATH_FILTER" ]]; then
    [[ "$f" == "$PATH_FILTER" || "$f" == "$PATH_FILTER"/* ]] || continue
  fi
  if is_cpp_file "$f"; then
    files+=("$f")
  fi
done

if [[ ${#files[@]} -eq 0 ]]; then
  if [[ -n "$PATH_FILTER" ]]; then
    echo "[format] no C/C++ files to check (${MODE} mode, path=${PATH_FILTER})"
  else
    echo "[format] no C/C++ files to check (${MODE} mode)"
  fi
  exit 0
fi

if ! command -v clang-format >/dev/null 2>&1; then
  echo "ERROR: clang-format is required. Install clang-format and rerun." >&2
  exit 1
fi

if [[ -n "$PATH_FILTER" ]]; then
  echo "[format] checking ${#files[@]} files (${MODE} mode, path=${PATH_FILTER})"
else
  echo "[format] checking ${#files[@]} files (${MODE} mode)"
fi
failed=0
for f in "${files[@]}"; do
  if ! clang-format --dry-run --Werror "$f" >/dev/null 2>&1; then
    echo "[format] needs formatting: $f"
    if [[ "$VERBOSE" -eq 1 ]]; then
      echo "[format] diff for $f:"
      clang-format "$f" | diff -u "$f" - || true
    fi
    failed=1
  fi
done

if [[ $failed -ne 0 ]]; then
  echo "[format] failed. Run clang-format on the files above." >&2
  exit 1
fi

echo "[format] OK"
