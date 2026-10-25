#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

ensure_card_sima_cli() {
  if ssh_card 'export PATH="/data/sima-cli/.venv/bin:${HOME}/.sima-cli/.venv/bin:${PATH}"; command -v sima-cli >/dev/null 2>&1'; then
    return
  fi

  local installer="${WORKSPACE}/scripts/ci/install_sima_cli_main.sh"
  if [[ ! -f "${installer}" ]]; then
    echo "ERROR: unable to bootstrap sima-cli on card; missing ${installer}" >&2
    exit 1
  fi

  echo "Installing sima-cli on PCIe card"
  local remote_installer="/tmp/sima-neat-install-sima-cli-main.sh"
  ssh_card "cat > $(shell_quote "${remote_installer}")" <"${installer}"
  ssh_card \
    "REMOTE_INSTALLER=$(shell_quote "${remote_installer}") \
     DEVKIT_PASSWORD=$(shell_quote "${DEVKIT_PASSWORD:-}") \
     SUDO_PASSWORD=$(shell_quote "${DEVKIT_PASSWORD:-}") \
     bash -s" <<REMOTE_BOOTSTRAP
set -euo pipefail
$(remote_sudo_wrapper_script)
setup_remote_sudo_wrapper
trap 'rm -rf "\${REMOTE_SUDO_WRAPPER_DIR:-}" "\${REMOTE_INSTALLER}"' EXIT
bash "\${REMOTE_INSTALLER}"
REMOTE_BOOTSTRAP
}

if [[ -n "${SIMAPCIE_LOCAL_CORE_DIST:-}" ]]; then
  metadata_sha256="$(sha256sum "${SIMAPCIE_LOCAL_CORE_DIST}/metadata-all.json")"
  metadata_sha256="${metadata_sha256%% *}"
  "${WORKSPACE}/scripts/ci/verify_core_dist.sh" \
    --dist "${SIMAPCIE_LOCAL_CORE_DIST}" \
    --expected-manifest "${WORKSPACE}/deps/manifest.json" \
    --expected-commit "${GITHUB_SHA:?}" \
    --package-profile core-codecs-llima
  remote_dist="$(ssh_card 'mkdir -p "${HOME}/tmp"; mktemp -d "${HOME}/tmp/sima-neat-card-validation.XXXXXX"')"
  [[ "${remote_dist}" =~ ^/[^[:space:]]*/sima-neat-card-validation\.[[:alnum:]]{6}$ ]] || {
    echo 'ERROR: invalid PCIe card candidate directory' >&2
    exit 1
  }
  cleanup_validation_candidate() {
    ssh_card "rm -rf -- $(shell_quote "${remote_dist}")" || true
  }
  trap cleanup_validation_candidate EXIT
  tar -C "${SIMAPCIE_LOCAL_CORE_DIST}" -cf - . | \
    ssh_card "tar -xf - -C $(shell_quote "${remote_dist}")"
  ssh_card \
    "REMOTE_DIST=$(shell_quote "${remote_dist}") \
     REMOTE_METADATA_SHA256=$(shell_quote "${metadata_sha256}") \
     DEVKIT_PASSWORD=$(shell_quote "${DEVKIT_PASSWORD:-}") \
     SUDO_PASSWORD=$(shell_quote "${DEVKIT_PASSWORD:-}") \
     bash -s" <<REMOTE_LOCAL_INSTALL
set -euo pipefail
$(remote_sudo_wrapper_script)
setup_remote_sudo_wrapper
trap 'rm -rf "\${REMOTE_SUDO_WRAPPER_DIR:-}"' EXIT
cd "\${REMOTE_DIST}"
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

metadata_bytes = Path('metadata-all.json').read_bytes()
if hashlib.sha256(metadata_bytes).hexdigest() != os.environ['REMOTE_METADATA_SHA256']:
    raise SystemExit('Transferred Core candidate metadata differs from verified input')
metadata = json.loads(metadata_bytes)
for name, expected in metadata['resources-checksum'].items():
    path = Path(name)
    if (path.name != name or path.is_symlink() or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
        raise SystemExit('Transferred Core candidate resource differs: ' + name)
PY
export NEAT_INSTALLER_SENTINEL_REQUIRED=ON
bash ./install_neat_framework.sh --local
REMOTE_LOCAL_INSTALL
  exit 0
fi

package_spec="core@${REF_NAME}:${SHORT_SHA}"
if [[ -n "${SIMAPCIE_CARD_INSTALL_CMD:-}" ]]; then
  echo "Installing card runtime with SIMAPCIE_CARD_INSTALL_CMD"
  export VULCAN_ENV REF_NAME SHORT_SHA CARD_HOST CARD_USER package_spec
  export DEVKIT_PASSWORD="${DEVKIT_PASSWORD:-}"
  export SUDO_PASSWORD="${DEVKIT_PASSWORD:-}"
  export NEAT_INSTALLER_SENTINEL_REQUIRED=ON
  bash -lc "${SIMAPCIE_CARD_INSTALL_CMD}"
  exit 0
fi

ensure_card_sima_cli

echo "Installing card runtime ${package_spec} on ${CARD_USER}@${CARD_HOST} from Vulcan env ${VULCAN_ENV}"
ssh_card \
  "REMOTE_VULCAN_ENV=$(shell_quote "${VULCAN_ENV}") \
   REMOTE_PACKAGE_SPEC=$(shell_quote "${package_spec}") \
   REMOTE_CARD_INSTALL_DIR=$(shell_quote "${SIMAPCIE_CARD_INSTALL_DIR:-}") \
   REMOTE_DEVKIT_PASSWORD=$(shell_quote "${DEVKIT_PASSWORD:-}") \
   REMOTE_SUDO_PASSWORD=$(shell_quote "${DEVKIT_PASSWORD:-}") \
   bash -s" <<'REMOTE_INSTALL'
set -euo pipefail

export PATH="/data/sima-cli/.venv/bin:${HOME}/.sima-cli/.venv/bin:${PATH}"
if ! command -v sima-cli >/dev/null 2>&1; then
  echo "ERROR: sima-cli is not installed on the PCIe card after bootstrap." >&2
  exit 1
fi

card_install_dir="${REMOTE_CARD_INSTALL_DIR}"
cleanup_card_install_dir=""
if [[ -z "${card_install_dir}" ]]; then
  card_install_root="${HOME}/tmp"
  mkdir -p "${card_install_root}"
  card_install_dir="$(mktemp -d "${card_install_root}/sima-neat-card-install.XXXXXX")"
  cleanup_card_install_dir="${card_install_dir}"
else
  mkdir -p "${card_install_dir}"
fi
echo "Using card installation directory ${card_install_dir}"

export SIMA_CLI_CHECK_FOR_UPDATE=0
export DEVKIT_PASSWORD="${REMOTE_DEVKIT_PASSWORD}"
export SUDO_PASSWORD="${REMOTE_SUDO_PASSWORD}"
# Benchmark telemetry is part of CI; fail when Sentinel cannot be installed.
export NEAT_INSTALLER_SENTINEL_REQUIRED=ON
setup_remote_sudo_wrapper() {
  local password="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
  if [[ -z "${password}" || ! -x /usr/bin/sudo ]]; then
    return 0
  fi

  REMOTE_SUDO_WRAPPER_DIR="$(mktemp -d /tmp/sima-neat-sudo-wrapper.XXXXXX)"
  export REMOTE_SUDO_WRAPPER_DIR
  cat > "${REMOTE_SUDO_WRAPPER_DIR}/sudo" <<'SUDO_WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
real_sudo="/usr/bin/sudo"
password="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
if "${real_sudo}" -n true 2>/dev/null; then
  exec "${real_sudo}" "$@"
fi
if [[ -z "${password}" ]]; then
  exec "${real_sudo}" "$@"
fi
printf '%s\n' "${password}" | "${real_sudo}" -S -p '' "$@"
SUDO_WRAPPER
  chmod 0700 "${REMOTE_SUDO_WRAPPER_DIR}/sudo"
  export PATH="${REMOTE_SUDO_WRAPPER_DIR}:${PATH}"
}
setup_remote_sudo_wrapper
trap 'rm -rf "${REMOTE_SUDO_WRAPPER_DIR:-}" "${cleanup_card_install_dir:-}"' EXIT
cd "${card_install_dir}"
# PCIe cards expose the Modalix userspace as Debian to sima-cli. Let the
# package installer perform the authoritative /etc/buildinfo and runtime
# profile checks after bypassing that outer metadata classification.
sima-cli install --neat --force --env "${REMOTE_VULCAN_ENV}" "${REMOTE_PACKAGE_SPEC}" -t all
REMOTE_INSTALL
