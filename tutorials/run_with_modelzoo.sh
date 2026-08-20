#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <model-name> <expected-tar> <model-target-folder> <modelzoo-version> <executable> [args...]" >&2
  exit 2
fi

MODEL_NAME="$1"
EXPECTED_TAR="$2"
MODEL_TARGET_FOLDER="$3"
MODELZOO_VERSION="$4"
EXE="$5"
shift 5

is_file() {
  [[ -f "$1" && -s "$1" ]]
}

find_sima_cli() {
  if [[ -n "${SIMA_CLI:-}" && -x "${SIMA_CLI}" ]]; then
    printf '%s\n' "${SIMA_CLI}"
    return 0
  fi
  if command -v sima-cli >/dev/null 2>&1; then
    command -v sima-cli
    return 0
  fi
  for candidate in /usr/bin/sima-cli /usr/local/bin/sima-cli /opt/sima/bin/sima-cli; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

find_downloaded_tar() {
  local model_base="${MODEL_NAME##*/}"
  local model_dash="${model_base//_/-}"
  local model_underscore="${model_base//-/_}"
  local -a dirs=(
    "${MODEL_TARGET_FOLDER}"
    "${MODEL_TARGET_FOLDER}/tmp"
    "$(pwd)"
    "$(pwd)/tmp"
    "${HOME:-}/.simaai"
    "${HOME:-}/.simaai/modelzoo"
    "${HOME:-}/.sima/modelzoo"
    "/data/simaai/modelzoo"
  )
  local -a names=(
    "${model_base}_mpk.tar.gz"
    "${model_dash}_mpk.tar.gz"
    "${model_underscore}_mpk.tar.gz"
    "${model_base}.tar.gz"
  )
  local dir name candidate
  for dir in "${dirs[@]}"; do
    [[ -n "${dir}" ]] || continue
    for name in "${names[@]}"; do
      candidate="${dir}/${name}"
      if is_file "${candidate}"; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    done
  done
  return 1
}

copy_to_expected() {
  local src="$1"
  if [[ "$(readlink -f "${src}")" != "$(readlink -m "${EXPECTED_TAR}")" ]]; then
    cp -f "${src}" "${EXPECTED_TAR}"
  fi
}

if ! is_file "${EXPECTED_TAR}"; then
  mkdir -p "$(dirname "${EXPECTED_TAR}")" "${MODEL_TARGET_FOLDER}"

  if found="$(find_downloaded_tar)"; then
    copy_to_expected "${found}"
  else
    if ! sima_cli="$(find_sima_cli)"; then
      echo "Missing ${EXPECTED_TAR} and cannot find sima-cli to download ${MODEL_NAME}" >&2
      exit 1
    fi
    echo "Missing ${EXPECTED_TAR}; downloading ${MODEL_NAME} from Model Zoo ${MODELZOO_VERSION}" >&2
    (
      cd "${MODEL_TARGET_FOLDER}"
      SIMA_CLI_CHECK_FOR_UPDATE=0 \
        "${sima_cli}" modelzoo -v "${MODELZOO_VERSION}" get "${MODEL_NAME}"
    )
    if found="$(find_downloaded_tar)"; then
      copy_to_expected "${found}"
    fi
  fi
fi

if ! is_file "${EXPECTED_TAR}"; then
  echo "Unable to resolve ${MODEL_NAME} tarball at ${EXPECTED_TAR}" >&2
  exit 1
fi

exec "${EXE}" "$@"
