#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: tools/prepare_s3_artifacts.sh \
  --core-deb <path> \
  --extras-tar <path> \
  --wheel <path> \
  --internals-manifest <path> \
  --output-dir <path> \
  [--internals-base-url <url>]
USAGE
}

CORE_DEB=""
EXTRAS_TAR=""
WHEEL_PATH=""
INTERNALS_MANIFEST=""
OUTPUT_DIR=""
INTERNALS_BASE_URL="${NEAT_INTERNALS_BASE_URL:-https://neat-artifacts.modalix.info/neat-internals}"
INSTALL_SCRIPT_PATH="tools/install_neat_framework.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --core-deb)
      CORE_DEB="${2:-}"
      shift 2
      ;;
    --extras-tar)
      EXTRAS_TAR="${2:-}"
      shift 2
      ;;
    --wheel)
      WHEEL_PATH="${2:-}"
      shift 2
      ;;
    --internals-manifest)
      INTERNALS_MANIFEST="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --internals-base-url)
      INTERNALS_BASE_URL="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${CORE_DEB}" || -z "${EXTRAS_TAR}" || -z "${WHEEL_PATH}" || -z "${INTERNALS_MANIFEST}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

[[ -f "${CORE_DEB}" ]] || { echo "Missing core deb: ${CORE_DEB}" >&2; exit 1; }
[[ -f "${EXTRAS_TAR}" ]] || { echo "Missing extras tar: ${EXTRAS_TAR}" >&2; exit 1; }
[[ -f "${WHEEL_PATH}" ]] || { echo "Missing wheel: ${WHEEL_PATH}" >&2; exit 1; }
[[ -f "${INTERNALS_MANIFEST}" ]] || { echo "Missing internals manifest: ${INTERNALS_MANIFEST}" >&2; exit 1; }
[[ -f "${INSTALL_SCRIPT_PATH}" ]] || { echo "Missing install script: ${INSTALL_SCRIPT_PATH}" >&2; exit 1; }

tmp_dir="$(mktemp -d /tmp/sima-neat-upload-XXXXXX)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

extract_dir="${tmp_dir}/internals-extract"
mkdir -p "${extract_dir}"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

cp "${CORE_DEB}" "${OUTPUT_DIR}/"
cp "${EXTRAS_TAR}" "${OUTPUT_DIR}/"
cp "${WHEEL_PATH}" "${OUTPUT_DIR}/"
cp "${INSTALL_SCRIPT_PATH}" "${OUTPUT_DIR}/"
chmod +x "${OUTPUT_DIR}/$(basename "${INSTALL_SCRIPT_PATH}")"

INTERNALS_TAG="$(python3 - <<'PY' "${INTERNALS_MANIFEST}"
import json
import sys
from pathlib import Path
manifest = Path(sys.argv[1])
data = json.loads(manifest.read_text(encoding="utf-8"))
tag = str(data.get("artifact_tag", "")).strip()
if not tag:
    raise SystemExit("artifact_tag missing in neat-internals manifest")
print(tag)
PY
)"

INTERNALS_ARCHIVE="sima-neat-internals-${INTERNALS_TAG}.tar.gz"
INTERNALS_ARCHIVE_PATH="${tmp_dir}/${INTERNALS_ARCHIVE}"
curl -fsSL "${INTERNALS_BASE_URL}/${INTERNALS_ARCHIVE}" -o "${INTERNALS_ARCHIVE_PATH}"
tar -xzf "${INTERNALS_ARCHIVE_PATH}" -C "${extract_dir}"

mapfile -t INTERNALS_DEBS < <(find "${extract_dir}" -type f -name '*.deb' | sort)
if [[ "${#INTERNALS_DEBS[@]}" -eq 0 ]]; then
  echo "No neat-internals .deb files found in ${INTERNALS_ARCHIVE}." >&2
  exit 1
fi
for f in "${INTERNALS_DEBS[@]}"; do
  cp "${f}" "${OUTPUT_DIR}/"
done

python3 tools/generate_sima_cli_metadata.py \
  --artifacts-dir "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/metadata.json"

echo "Prepared upload artifacts:"
ls -lh "${OUTPUT_DIR}"
