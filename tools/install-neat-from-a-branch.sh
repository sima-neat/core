#!/usr/bin/env bash
set -euo pipefail

# install-neat-from-a-branch.sh
#
# Purpose:
# - Install NEAT from R2-hosted metadata via sima-cli.
# - If a ref is not provided, fetches branches.json and prompts user to choose
#   from available branches and releases.
#
# This is for users to easily install NEAT from a specific branch, release, or
# branch build hash without needing to know exact URLs.
#
# Inputs:
# - Optional flag: -minimum / --minimum
#   - install from metadata-minimal.json instead of metadata.json
# - Optional flag: --all
#   - install from metadata-all.json so extras are pulled without prompting
# - Positional arg 1 (optional): branch or release name
# - Positional arg 2 (optional): artifact tag or git hash
#   - latest: resolve latest.tag for the selected branch
#   - for releases, latest resolves to the selected release
#   - otherwise: use the provided value directly
#
# Environment:
# - NEAT_ARTIFACTS_BASE_URL: base URL for artifact index/tag metadata
#   (default: https://artifacts.sima-neat.com/core)
# - SIMA_CLI_BIN: sima-cli binary/path override (default: sima-cli)
#
# Example:
# - Non-interactive latest branch:  bash tools/install-neat-from-a-branch.sh feature/docs
# - Non-interactive fixed branch:   bash tools/install-neat-from-a-branch.sh feature/docs 1a2b3c4
# - Non-interactive release:        bash tools/install-neat-from-a-branch.sh v0.0.1
# - Minimal install latest:         bash tools/install-neat-from-a-branch.sh -minimum feature/docs
# - All-components latest:          bash tools/install-neat-from-a-branch.sh --all feature/docs
# - Interactive:             bash tools/install-neat-from-a-branch.sh
#
BASE_URL="${NEAT_ARTIFACTS_BASE_URL:-https://artifacts.sima-neat.com/core}"
CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"
CLI_FALLBACK="/data/sima-cli/.venv/bin/sima-cli"
MINIMUM=0
INSTALL_ALL=0
REF_NAME=""
TAG_INPUT="latest"
METADATA_FILE="metadata.json"
REF_TYPE="branch"
CATALOG_JSON=""
CATALOG_LOADED=0
BRANCHES=()
RELEASES=()

usage() {
  cat <<'USAGE'
Usage:
  install-neat-from-a-branch.sh [-minimum|--minimum] [--all] [branch-or-release] [latest|git-hash]

Environment:
  NEAT_ARTIFACTS_BASE_URL  Base URL for neat artifacts (default: https://artifacts.sima-neat.com/core)
  SIMA_CLI_BIN             sima-cli executable name/path (default: sima-cli)
USAGE
}

download_text() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -qO- "${url}"
    return 0
  fi
  echo "Neither curl nor wget is installed." >&2
  return 1
}

sanitize_branch_key() {
  printf '%s' "$1" | tr '/ ' '--'
}

url_exists() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsI "${url}" >/dev/null 2>&1
    return $?
  fi
  if command -v wget >/dev/null 2>&1; then
    wget --spider -q "${url}" >/dev/null 2>&1
    return $?
  fi
  echo "Neither curl nor wget is installed." >&2
  return 1
}

load_catalog() {
  if [[ "${CATALOG_LOADED}" -eq 1 ]]; then
    return 0
  fi

  CATALOG_JSON="$(download_text "${BASE_URL}/branches.json")"

  mapfile -t BRANCHES < <(python3 - <<'PY' "${CATALOG_JSON}"
import json
import sys

raw = sys.argv[1]
data = json.loads(raw)
values = data.get("branches", []) if isinstance(data, dict) else data if isinstance(data, list) else []
for item in values:
    s = str(item).strip()
    if s:
        print(s)
PY
)

  mapfile -t RELEASES < <(python3 - <<'PY' "${CATALOG_JSON}"
import json
import sys

raw = sys.argv[1]
data = json.loads(raw)
values = data.get("releases", []) if isinstance(data, dict) else []
for item in values:
    s = str(item).strip()
    if s:
        print(s)
PY
)

  CATALOG_LOADED=1
}

contains_value() {
  local needle="$1"
  shift || true
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

detect_ref_type() {
  local ref_name="$1"

  load_catalog || true

  if contains_value "${ref_name}" "${RELEASES[@]}"; then
    printf '%s\n' "release"
    return 0
  fi
  if contains_value "${ref_name}" "${BRANCHES[@]}"; then
    printf '%s\n' "branch"
    return 0
  fi
  if [[ "${ref_name}" =~ ^v[0-9] ]]; then
    printf '%s\n' "release"
    return 0
  fi
  printf '%s\n' "branch"
}

prompt_for_ref() {
  local -a entries=()
  local -a priority_branches=()
  local -a other_branches=()
  local divider_index=0
  local branch
  local release
  local i
  local choice

  load_catalog

  for release in "${RELEASES[@]}"; do
    entries+=("release:${release}")
  done

  for branch in "${BRANCHES[@]}"; do
    case "${branch}" in
      main|develop)
        priority_branches+=("${branch}")
        ;;
      *)
        other_branches+=("${branch}")
        ;;
    esac
  done

  if [[ " ${priority_branches[*]} " == *" main "* ]]; then
    entries+=("branch:main")
  fi
  if [[ " ${priority_branches[*]} " == *" develop "* ]]; then
    entries+=("branch:develop")
  fi
  if [[ "${#other_branches[@]}" -gt 0 && "${#entries[@]}" -gt 0 ]]; then
    divider_index="${#entries[@]}"
  fi
  for branch in "${other_branches[@]}"; do
    entries+=("branch:${branch}")
  done

  if [[ "${#entries[@]}" -eq 0 ]]; then
    echo "No branches or releases found in ${BASE_URL}/branches.json." >&2
    exit 1
  fi

  echo "Available refs:"
  for i in "${!entries[@]}"; do
    if [[ "${divider_index}" -gt 0 && "${i}" -eq "${divider_index}" ]]; then
      printf "     %s\n" "------------------------------"
    fi
    local kind="${entries[$i]%%:*}"
    local value="${entries[$i]#*:}"
    if [[ "${kind}" == "release" ]]; then
      printf "  %2d) [release] %s\n" "$((i + 1))" "${value}"
    else
      printf "  %2d) [branch]  %s\n" "$((i + 1))" "${value}"
    fi
  done

  read -r -p "Choose ref [1-${#entries[@]}]: " choice
  if [[ ! "${choice}" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#entries[@]} )); then
    echo "Invalid selection: ${choice}" >&2
    exit 1
  fi

  REF_TYPE="${entries[$((choice - 1))]%%:*}"
  REF_NAME="${entries[$((choice - 1))]#*:}"
}

resolve_release_metadata_url() {
  local release_name="$1"
  local release_key
  local tag_content
  local resolved_tag

  if [[ "${TAG_INPUT}" != "latest" ]]; then
    echo "Release installs do not accept an extra git hash/tag argument. Use '${release_name}' by itself." >&2
    exit 1
  fi

  release_key="$(sanitize_branch_key "${release_name}")"
  tag_content="$(download_text "${BASE_URL}/${release_key}/latest.tag" || true)"
  resolved_tag="$(printf '%s' "${tag_content}" | tr -d '[:space:]')"
  if [[ -z "${resolved_tag}" ]]; then
    echo "latest.tag is empty or unavailable for release: ${release_name} (key: ${release_key})" >&2
    exit 1
  fi

  printf '%s\n' "${BASE_URL}/${release_key}/${resolved_tag}/${METADATA_FILE}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -minimum|--minimum)
      MINIMUM=1
      shift
      ;;
    --all)
      INSTALL_ALL=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  REF_NAME="$1"
  shift
fi

if [[ $# -gt 0 ]]; then
  TAG_INPUT="$1"
  shift
fi

if [[ $# -gt 0 ]]; then
  echo "Unexpected argument: $1" >&2
  usage
  exit 1
fi

if [[ "${MINIMUM}" -eq 1 && "${INSTALL_ALL}" -eq 1 ]]; then
  echo "--minimum and --all cannot be used together." >&2
  usage
  exit 1
fi

if [[ "${MINIMUM}" -eq 1 ]]; then
  METADATA_FILE="metadata-minimal.json"
elif [[ "${INSTALL_ALL}" -eq 1 ]]; then
  METADATA_FILE="metadata-all.json"
fi

if ! command -v "${CLI_BIN}" >/dev/null 2>&1; then
  if [[ "${CLI_BIN}" == "sima-cli" && -x "${CLI_FALLBACK}" ]]; then
    CLI_BIN="${CLI_FALLBACK}"
  else
    echo "Required command not found: ${CLI_BIN}" >&2
    if [[ "${CLI_BIN}" == "sima-cli" ]]; then
      echo "Also checked fallback path: ${CLI_FALLBACK}" >&2
    fi
    exit 1
  fi
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to parse branches.json." >&2
  exit 1
fi

if [[ -z "${REF_NAME}" ]]; then
  prompt_for_ref
else
  REF_TYPE="$(detect_ref_type "${REF_NAME}")"
fi

if [[ "${REF_TYPE}" == "release" ]]; then
  METADATA_URL="$(resolve_release_metadata_url "${REF_NAME}")"
else
  BRANCH_KEY="$(sanitize_branch_key "${REF_NAME}")"

  if [[ "${TAG_INPUT}" == "latest" ]]; then
    TAG_CONTENT="$(download_text "${BASE_URL}/${BRANCH_KEY}/latest.tag" || true)"
    TAG="$(printf '%s' "${TAG_CONTENT}" | tr -d '[:space:]')"
    if [[ -z "${TAG}" ]]; then
      echo "latest.tag is empty or unavailable for branch: ${REF_NAME} (key: ${BRANCH_KEY})" >&2
      exit 1
    fi
  else
    TAG="${TAG_INPUT}"
  fi

  METADATA_URL="${BASE_URL}/${BRANCH_KEY}/${TAG}/${METADATA_FILE}"
fi

echo "Installing from ${METADATA_URL}"
"${CLI_BIN}" install -m "${METADATA_URL}"
