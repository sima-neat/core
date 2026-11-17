#!/usr/bin/env bash
set -euo pipefail

resolve_sima_cli_python() {
  local vulcan_cli="${1:-/mnt/vulcan/bin/sima-cli}"
  local cli python bin_dir cache_venv
  if [[ -n "${SIMA_CLI_PYTHON:-}" ]]; then
    if ! python="$(command -v "${SIMA_CLI_PYTHON}")" || [[ ! -f "${python}" || ! -x "${python}" ]]; then
      echo "Cannot execute explicit SIMA_CLI_PYTHON: ${SIMA_CLI_PYTHON}" >&2
      return 1
    fi
    printf '%s\n' "${python}"
    return 0
  fi

  if [[ -n "${SIMA_CLI_BIN:-}" ]]; then
    if ! cli="$(command -v "${SIMA_CLI_BIN}")"; then
      echo "Cannot locate explicit SIMA_CLI_BIN: ${SIMA_CLI_BIN}" >&2
      return 1
    fi
  elif cli="$(command -v sima-cli)"; then
    :
  elif [[ -x "${vulcan_cli}" ]]; then
    cli="${vulcan_cli}"
  else
    echo "Cannot locate sima-cli on PATH or at ${vulcan_cli}; set SIMA_CLI_BIN or SIMA_CLI_PYTHON." >&2
    return 1
  fi
  if ! cli="$(readlink -f -- "${cli}")" || [[ ! -f "${cli}" || ! -x "${cli}" ]]; then
    echo "Selected sima-cli is not an executable file; set SIMA_CLI_BIN or SIMA_CLI_PYTHON." >&2
    return 1
  fi
  bin_dir="$(dirname "${cli}")"
  cache_venv="${bin_dir}/../.sima-cli/.venv"
  if [[ -f "${bin_dir}/../pyvenv.cfg" && -x "${bin_dir}/python" ]]; then
    python="${bin_dir}/python"
  elif [[ "${cli}" == */bin/sima-cli && -f "${cache_venv}/pyvenv.cfg" &&
          -x "${cache_venv}/bin/python" && -f "${cache_venv}/bin/sima-cli" &&
          -x "${cache_venv}/bin/sima-cli" ]]; then
    # Vulcan's portable cache launcher executes this path-relative venv, not
    # an interpreter beside the wrapper. Do not fall back to ambient Python.
    python="${cache_venv}/bin/python"
  else
    echo "Cannot locate the Python belonging to selected sima-cli ${cli}; set SIMA_CLI_PYTHON explicitly." >&2
    return 1
  fi
  # Keep the venv path: resolving its Python symlink would lose that environment.
  printf '%s\n' "${python}"
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="$(resolve_sima_cli_python)"
exec "${python}" "${root}/tools/ci/verify_core_dist.py" "$@"
