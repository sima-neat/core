#!/usr/bin/env bash
set -euo pipefail

base_url="${SIMA_CLI_ARTIFACT_BASE_URL:-https://artifacts.neat.sima.ai/sima-cli}"
ref="${SIMA_CLI_ARTIFACT_REF:-main}"
tag="${SIMA_CLI_ARTIFACT_TAG:-latest}"
ref_key="${ref//\//-}"
ref_key="${ref_key// /-}"

if [[ "${tag}" == "latest" ]]; then
  tag="$(curl -fsSL "${base_url}/${ref_key}/latest.tag" | tr -d '[:space:]')"
fi

if [[ -z "${tag}" ]]; then
  echo "ERROR: unable to resolve sima-cli artifact tag for ${ref}" >&2
  exit 1
fi

metadata_url="${base_url}/${ref_key}/${tag}/metadata.json"
tmp_dir="$(mktemp -d /tmp/sima-cli-artifact-install.XXXXXX)"
trap 'rm -rf "${tmp_dir}"' EXIT

echo "Installing sima-cli from ${metadata_url}"
curl -fsSL "${metadata_url}" -o "${tmp_dir}/metadata.json"

while IFS=$'\t' read -r resource resource_url; do
  [[ -n "${resource}" && -n "${resource_url}" ]] || continue
  output="${tmp_dir}/${resource}"
  mkdir -p "$(dirname "${output}")"
  echo "Downloading ${resource_url}"
  curl -fsSL "${resource_url}" -o "${output}"
done < <(python3 - "${metadata_url}" "${tmp_dir}/metadata.json" <<'PY'
import json
import sys
import urllib.parse

metadata_url = sys.argv[1]
metadata_path = sys.argv[2]
base_url = metadata_url.rsplit("/", 1)[0]

with open(metadata_path, encoding="utf-8") as f:
    metadata = json.load(f)

resources = metadata.get("resources", [])
if not isinstance(resources, list) or not resources:
    raise SystemExit(f"No resources listed in {metadata_url}")

for resource in resources:
    if not isinstance(resource, str) or not resource:
        continue
    resource_url = f"{base_url}/{urllib.parse.quote(resource)}"
    print(f"{resource}\t{resource_url}")
PY
)

(
  cd "${tmp_dir}"
  python3 install_vulcan_package.py
)

if [[ -n "${SIMA_CLI_VERIFIER_PYTHON:-}" ]]; then
  # Keep the host ABI interpreter while verifying firmware with fixed tar filters.
  verifier_venv="${RUNNER_TEMP:?}/core-validation-cli"
  "${SIMA_CLI_VERIFIER_PYTHON}" -m venv "${verifier_venv}"
  "${verifier_venv}/bin/python" - "${tmp_dir}" <<'PY'
from pathlib import Path
import subprocess
import sys
import zipfile

root = Path(sys.argv[1])
packages = list(root.glob('sima-cli-package-*.zip'))
if len(packages) != 1:
    raise SystemExit('Expected one downloaded sima-cli package')
with zipfile.ZipFile(packages[0]) as archive:
    wheels = [name for name in archive.namelist()
              if name.endswith('.whl') and Path(name).name == name]
    if len(wheels) != 1:
        raise SystemExit('Expected one published sima-cli wheel')
    wheel = root / wheels[0]
    wheel.write_bytes(archive.read(wheels[0]))
subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                '--no-cache-dir', str(wheel)], check=True)
PY
  printf 'SIMA_CLI_PYTHON=%s\n' "${verifier_venv}/bin/python" >> "${GITHUB_ENV:?}"
fi
