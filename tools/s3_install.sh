#!/usr/bin/env bash
set -euo pipefail

# s3_install.sh
#
# Purpose:
# - Install NEAT from S3-hosted metadata via sima-cli.
# - If branch is not provided, fetches branches.json and prompts user to choose.
#
# This is for user to easily install NEAT from a specific branch/tag without needing to know exact URLs.
#
# Inputs:
# - Positional arg 1 (optional): branch name
#
# Environment:
# - NEAT_ARTIFACTS_BASE_URL: base URL for artifact index/tag metadata
#   (default: https://neat-artifacts.modalix.info/neat)
# - SIMA_CLI_BIN: sima-cli binary/path override (default: sima-cli)
#
# Example:
# - Non-interactive: bash tools/s3_install.sh feature/docs
# - Interactive:     bash tools/s3_install.sh
#
BASE_URL="${NEAT_ARTIFACTS_BASE_URL:-https://neat-artifacts.modalix.info/neat}"
CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"
BRANCH="${1:-}"

usage() {
  cat <<'USAGE'
Usage:
  install.sh [branch]

Environment:
  NEAT_ARTIFACTS_BASE_URL  Base URL for neat artifacts (default: https://neat-artifacts.modalix.info/neat)
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

if [[ "${BRANCH}" == "-h" || "${BRANCH}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v "${CLI_BIN}" >/dev/null 2>&1; then
  echo "Required command not found: ${CLI_BIN}" >&2
  exit 1
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

TAG="$(download_text "${BASE_URL}/${BRANCH}/latest.tag" | tr -d '[:space:]')"
if [[ -z "${TAG}" ]]; then
  echo "latest.tag is empty for branch: ${BRANCH}" >&2
  exit 1
fi

METADATA_URL="${BASE_URL}/${BRANCH}/${TAG}/metadata.json"
echo "Installing from ${METADATA_URL}"
"${CLI_BIN}" install -m "${METADATA_URL}"
