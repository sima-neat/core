#!/usr/bin/env bash

simapcie_detect_host_distro() {
  local os_release_file="${1:-/etc/os-release}"
  if [[ ! -r "${os_release_file}" ]]; then
    echo "ERROR: cannot read ${os_release_file} to determine the host distribution" >&2
    return 1
  fi

  local os_id
  local os_version
  os_id="$(awk -F= '$1 == "ID" { value = substr($0, index($0, "=") + 1); gsub(/^"|"$/, "", value); print value; exit }' "${os_release_file}")"
  os_version="$(awk -F= '$1 == "VERSION_ID" { value = substr($0, index($0, "=") + 1); gsub(/^"|"$/, "", value); print value; exit }' "${os_release_file}")"

  case "${os_id}:${os_version}" in
    ubuntu:22.04)
      echo "ubuntu22"
      ;;
    ubuntu:24.04)
      echo "ubuntu24"
      ;;
    *)
      echo "ERROR: unsupported PCIe host distribution '${os_id:-unknown} ${os_version:-unknown}'." >&2
      echo "       Supported distributions are Ubuntu 22.04 and Ubuntu 24.04." >&2
      return 1
      ;;
  esac
}
