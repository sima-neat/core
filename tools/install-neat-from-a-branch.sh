#!/usr/bin/env bash
set -euo pipefail

# install-neat-from-a-branch.sh
#
# Purpose:
# - Install NEAT from S3-hosted metadata via sima-cli.
# - If branch is not provided, fetches branches.json and prompts user to choose.
#
# This is for user to easily install NEAT from a specific branch/tag without needing to know exact URLs.
#
# Inputs:
# - Optional flag: -minimum / --minimum
#   - install from metadata-minimal.json instead of metadata.json
# - Optional flag: --all
#   - install from metadata-all.json so extras are pulled without prompting
# - Positional arg 1 (optional): branch name
# - Positional arg 2 (optional): artifact tag or git hash
#   - latest: resolve latest.tag for the selected branch
#   - otherwise: use the provided value directly
#
# Environment:
# - NEAT_ARTIFACTS_BASE_URL: base URL for artifact index/tag metadata
#   (default: https://sima-neat.com/core)
# - SIMA_CLI_BIN: sima-cli binary/path override (default: sima-cli)
#
# Example:
# - Non-interactive latest:  bash tools/install-neat-from-a-branch.sh feature/docs
# - Non-interactive fixed:   bash tools/install-neat-from-a-branch.sh feature/docs 1a2b3c4
# - Minimal install latest:  bash tools/install-neat-from-a-branch.sh -minimum feature/docs
# - All-components latest:   bash tools/install-neat-from-a-branch.sh --all feature/docs
# - Interactive:             bash tools/install-neat-from-a-branch.sh
#
BASE_URL="${NEAT_ARTIFACTS_BASE_URL:-https://sima-neat.com/core}"
CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"
CLI_FALLBACK="/data/sima-cli/.venv/bin/sima-cli"
MINIMUM=0
INSTALL_ALL=0
BRANCH=""
TAG_INPUT="latest"
METADATA_FILE="metadata.json"

usage() {
  cat <<'USAGE'
Usage:
  install-neat-from-a-branch.sh [-minimum|--minimum] [--all] [branch] [latest|git-hash]

Environment:
  NEAT_ARTIFACTS_BASE_URL  Base URL for neat artifacts (default: https://sima-neat.com/core)
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
  BRANCH="$1"
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

if [[ -z "${BRANCH}" ]]; then
  BRANCHES_JSON="$(download_text "${BASE_URL}/branches.json")"

  mapfile -t BRANCHES < <(python3 - <<'PY' "${BRANCHES_JSON}"
import json
import sys
raw = sys.argv[1]
data = json.loads(raw)
if isinstance(data, dict):
    values = data.get("branches", [])
elif isinstance(data, list):
    values = data
else:
    values = []
for item in values:
    s = str(item).strip()
    if s:
        print(s)
PY
)

  if [[ "${#BRANCHES[@]}" -eq 0 ]]; then
    echo "No branches found in ${BASE_URL}/branches.json." >&2
    exit 1
  fi

  echo "Available branches:"
  for i in "${!BRANCHES[@]}"; do
    printf "  %2d) %s\n" "$((i + 1))" "${BRANCHES[$i]}"
  done

  read -r -p "Choose branch [1-${#BRANCHES[@]}]: " choice
  if [[ ! "${choice}" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#BRANCHES[@]} )); then
    echo "Invalid selection: ${choice}" >&2
    exit 1
  fi
  BRANCH="${BRANCHES[$((choice - 1))]}"
fi

if [[ "${TAG_INPUT}" == "latest" ]]; then
  TAG="$(download_text "${BASE_URL}/${BRANCH}/latest.tag" | tr -d '[:space:]')"
  if [[ -z "${TAG}" ]]; then
    echo "latest.tag is empty for branch: ${BRANCH}" >&2
    exit 1
  fi
else
  TAG="${TAG_INPUT}"
fi

METADATA_URL="${BASE_URL}/${BRANCH}/${TAG}/${METADATA_FILE}"
echo "Installing from ${METADATA_URL}"
"${CLI_BIN}" install -m "${METADATA_URL}"
