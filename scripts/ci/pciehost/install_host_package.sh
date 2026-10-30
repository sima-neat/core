#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

host_distro="$(simapcie_detect_host_distro)"
if [[ -n "${SIMAPCIE_EXPECTED_HOST_DISTRO:-}" &&
  "${host_distro}" != "${SIMAPCIE_EXPECTED_HOST_DISTRO}" ]]; then
  echo "ERROR: detected host distro ${host_distro} does not match expected ${SIMAPCIE_EXPECTED_HOST_DISTRO}." >&2
  exit 1
fi

if [[ -n "${SIMAPCIE_LOCAL_HOST_DIST:-}" ]]; then
  python3 - "${WORKSPACE}" "${SIMAPCIE_LOCAL_HOST_DIST}" "${host_distro}" "${GITHUB_SHA:?}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

workspace, directory, distro, commit = sys.argv[1:]
root = Path(directory).resolve(strict=True)
metadata = json.loads((root / 'metadata.json').read_text())
if (metadata.get('commit') != commit or metadata.get('artifact', {}).get('distro') != distro
        or metadata.get('artifact', {}).get('arches') != ['amd64']):
    raise SystemExit('PCIe host candidate does not match this run and host')
resources = metadata['resources']
hashes = metadata['resources-checksum']
if not resources or len(resources) != len(set(resources)) or set(resources) != set(hashes):
    raise SystemExit('PCIe host candidate resource inventory differs')
for name in resources:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._+~-]*', name):
        raise SystemExit('Invalid PCIe host candidate resource')
    path = root / name
    if (path.is_symlink() or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]):
        raise SystemExit('PCIe host candidate resource differs: ' + name)
sys.path.insert(0, str(Path(workspace) / 'pcie_host/scripts'))
from pcie_delivery import verify, verify_publication_inventory
selection, _ = verify(root)
if selection['distro'] != distro or selection['architecture'] != 'amd64':
    raise SystemExit('Native Scheduler candidate does not match the host')
verify_publication_inventory(root, selection, resources)
PY
  mkdir -p "${PACKAGE_DIR}"
  cp -a "${SIMAPCIE_LOCAL_HOST_DIST}/." "${PACKAGE_DIR}/"
  SIMAPCIE_SSH_PASSWORD="${DEVKIT_PASSWORD:-}" \
  SIMAPCIE_SETUP_ARGS="--non-interactive --strict" \
    bash "${PACKAGE_DIR}/install_pciehost.sh" --search-dir "${PACKAGE_DIR}" --python
else
  package_spec="core/pciehost/${host_distro}/amd64@${REF_NAME}:${SHORT_SHA}"
  echo "Installing PCIe host package ${package_spec} from Vulcan env ${VULCAN_ENV}"
  SIMAPCIE_SSH_PASSWORD="${DEVKIT_PASSWORD:-}" \
  SIMAPCIE_SETUP_ARGS="--non-interactive --strict" \
  SIMA_CLI_CHECK_FOR_UPDATE=0 \
    sima-cli neat install \
      --env "${VULCAN_ENV}" \
      --install-dir "${PACKAGE_DIR}" \
      "${package_spec}"
fi

/usr/bin/gst-inspect-1.0 neatpciehost >/dev/null
echo "PCIe host package installed and neatpciehost is inspectable."
