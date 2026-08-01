#!/usr/bin/env bash

# This file is sourced by build.sh so its exported variables remain available
# to CMake and the sccache server.

simaneat_sccache_warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

simaneat_sccache_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    return 1
  fi
}

simaneat_install_sccache() {
  local version="${SIMANEAT_SCCACHE_VERSION:-v0.16.0}"
  local cache_home="${XDG_CACHE_HOME:-${HOME}/.cache}"
  local install_dir="${SIMANEAT_SCCACHE_INSTALL_DIR:-${cache_home}/sima-neat/tools/sccache/${version}}"
  local installed_bin="${install_dir}/sccache"

  if [[ -x "${installed_bin}" ]]; then
    printf '%s\n' "${installed_bin}"
    return 0
  fi

  local machine
  local os
  local target
  local expected_sha256
  machine="$(uname -m)"
  os="$(uname -s)"

  case "${os}:${machine}" in
    Linux:aarch64|Linux:arm64)
      target="aarch64-unknown-linux-musl"
      expected_sha256="f73a5c39f96bb6ebb89cc7915cf182260d4cbf30765322c5e793d0fe8bd80784"
      ;;
    Linux:x86_64|Linux:amd64)
      target="x86_64-unknown-linux-musl"
      expected_sha256="aec995a83ad3dff3d14b6314e08858b7b73d35ca85a5bcf3d3a9ec07dee35588"
      ;;
    Darwin:arm64|Darwin:aarch64)
      target="aarch64-apple-darwin"
      expected_sha256="ded590cae2c72042c61178632906bef62d635fa20d45f8b22110a2241f430960"
      ;;
    Darwin:x86_64|Darwin:amd64)
      target="x86_64-apple-darwin"
      expected_sha256="f7dbd055db75a938ab1539f5316c5d08e73a1b94c40ab170ddcc617f5bf18343"
      ;;
    *)
      simaneat_sccache_warn "No pinned sccache binary is available for ${os}/${machine}."
      return 1
      ;;
  esac

  command -v curl >/dev/null 2>&1 || {
    simaneat_sccache_warn "curl is required to bootstrap sccache."
    return 1
  }
  command -v tar >/dev/null 2>&1 || {
    simaneat_sccache_warn "tar is required to bootstrap sccache."
    return 1
  }

  local archive_name="sccache-${version}-${target}.tar.gz"
  local download_url="https://github.com/mozilla/sccache/releases/download/${version}/${archive_name}"
  local temp_dir
  local archive_path
  local extracted_bin
  local actual_sha256
  temp_dir="$(mktemp -d)"
  archive_path="${temp_dir}/${archive_name}"

  if ! curl --fail --location --silent --show-error "${download_url}" --output "${archive_path}"; then
    rm -rf "${temp_dir}"
    simaneat_sccache_warn "Failed to download sccache ${version}."
    return 1
  fi

  actual_sha256="$(simaneat_sccache_sha256 "${archive_path}")" || {
    rm -rf "${temp_dir}"
    simaneat_sccache_warn "sha256sum or shasum is required to verify sccache."
    return 1
  }
  if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    rm -rf "${temp_dir}"
    simaneat_sccache_warn "Checksum verification failed for ${archive_name}."
    return 1
  fi

  tar -xzf "${archive_path}" -C "${temp_dir}"
  extracted_bin="${temp_dir}/sccache-${version}-${target}/sccache"
  if [[ ! -x "${extracted_bin}" ]]; then
    rm -rf "${temp_dir}"
    simaneat_sccache_warn "Downloaded archive does not contain the expected sccache binary."
    return 1
  fi

  mkdir -p "${install_dir}"
  install -m 0755 "${extracted_bin}" "${installed_bin}"
  rm -rf "${temp_dir}"
  printf '%s\n' "${installed_bin}"
}

simaneat_configure_sccache() {
  local repo_root="$1"
  local requested_mode="${SIMANEAT_SCCACHE:-auto}"
  local normalized_mode
  local sccache_bin=""
  normalized_mode="$(printf '%s' "${requested_mode}" | tr '[:upper:]' '[:lower:]')"

  case "${normalized_mode}" in
    off|0|false|no)
      export SIMANEAT_SCCACHE_ACTIVE=OFF
      return 0
      ;;
    auto|on|1|true|yes)
      ;;
    *)
      echo "ERROR: SIMANEAT_SCCACHE must be auto, on, or off; got '${requested_mode}'." >&2
      return 1
      ;;
  esac

  if command -v sccache >/dev/null 2>&1; then
    sccache_bin="$(command -v sccache)"
  elif ! sccache_bin="$(simaneat_install_sccache)"; then
    if [[ "${normalized_mode}" == "auto" ]]; then
      simaneat_sccache_warn "Continuing without compiler caching. Set SIMANEAT_SCCACHE=off to suppress this warning."
      export SIMANEAT_SCCACHE_ACTIVE=OFF
      return 0
    fi
    echo "ERROR: sccache was requested but could not be installed." >&2
    return 1
  fi

  local cache_home="${XDG_CACHE_HOME:-${HOME}/.cache}"
  export SCCACHE_DIR="${SCCACHE_DIR:-${cache_home}/sima-neat/sccache}"
  export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-10G}"
  export SCCACHE_LOCAL_RW_MODE="${SCCACHE_LOCAL_RW_MODE:-READ_WRITE}"
  export SCCACHE_BASEDIRS="${SCCACHE_BASEDIRS:-${repo_root}}"

  if [[ -n "${SCCACHE_BUCKET:-}" ]]; then
    : "${SCCACHE_REGION:?SCCACHE_REGION is required when SCCACHE_BUCKET is set}"
    : "${SCCACHE_S3_KEY_PREFIX:?SCCACHE_S3_KEY_PREFIX is required when SCCACHE_BUCKET is set}"
    export SCCACHE_MULTILEVEL_CHAIN="${SCCACHE_MULTILEVEL_CHAIN:-disk,s3}"
    export SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY="${SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY:-l0}"
    export SCCACHE_S3_USE_SSL="${SCCACHE_S3_USE_SSL:-true}"
    export SCCACHE_S3_RW_MODE="${SCCACHE_S3_RW_MODE:-READ_ONLY}"
  fi

  mkdir -p "${SCCACHE_DIR}"
  export SIMANEAT_SCCACHE_BIN="${sccache_bin}"
  export SIMANEAT_SCCACHE_ACTIVE=ON

  if [[ "${SIMANEAT_SCCACHE_ZERO_STATS:-OFF}" == "ON" ]]; then
    if ! "${sccache_bin}" --zero-stats >/dev/null; then
      if [[ "${SIMANEAT_SCCACHE_STARTUP_FAIL_OPEN:-OFF}" == "ON" ]]; then
        simaneat_sccache_warn "sccache failed to start; continuing without compiler caching."
        export SIMANEAT_SCCACHE_ACTIVE=OFF
        return 0
      fi
      echo "ERROR: sccache failed to start." >&2
      return 1
    fi
  fi

  echo "sccache enabled: $("${sccache_bin}" --version)"
  echo "sccache local cache: ${SCCACHE_DIR} (${SCCACHE_CACHE_SIZE})"
  if [[ -n "${SCCACHE_BUCKET:-}" ]]; then
    echo "sccache remote cache: s3://${SCCACHE_BUCKET}/${SCCACHE_S3_KEY_PREFIX} (${SCCACHE_S3_RW_MODE})"
  fi
}

simaneat_show_sccache_stats() {
  if [[ "${SIMANEAT_SCCACHE_ACTIVE:-OFF}" == "ON" ]]; then
    echo
    echo "sccache statistics:"
    "${SIMANEAT_SCCACHE_BIN}" --show-stats
  fi
}
