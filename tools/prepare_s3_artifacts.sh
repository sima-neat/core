#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: tools/prepare_s3_artifacts.sh \
  --artifacts-dir <dist-dir> \
  --output-dir <path>

Legacy arguments such as --core-deb, --extras-tar, --wheel, and
--internals-manifest are accepted for compatibility but ignored. The complete
upload payload is built from --artifacts-dir.
USAGE
}

ARTIFACTS_DIR="dist"
OUTPUT_DIR=""
INSTALL_SCRIPT_NAME="${NEAT_PACKAGE_INSTALL_SCRIPT:-install_neat_framework.sh}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifacts-dir)
      ARTIFACTS_DIR="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --core-deb|--extras-tar|--wheel|--internals-manifest|--internals-deb-dir|--llima-deb-dir|--internals-base-url|--llima-base-url)
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

if [[ -z "${ARTIFACTS_DIR}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

[[ -d "${ARTIFACTS_DIR}" ]] || { echo "Missing artifacts dir: ${ARTIFACTS_DIR}" >&2; exit 1; }

CORE_DEBS=()
while IFS= read -r file; do
  CORE_DEBS+=("${file}")
done < <(find "${ARTIFACTS_DIR}" -maxdepth 1 -type f -name 'sima-neat-*-Linux-core.deb' | sort)

EXTRAS_TARS=()
while IFS= read -r file; do
  EXTRAS_TARS+=("${file}")
done < <(find "${ARTIFACTS_DIR}" -maxdepth 1 -type f -name '*extras.tar.gz' | sort)

WHEELS=()
while IFS= read -r file; do
  WHEELS+=("${file}")
done < <(find "${ARTIFACTS_DIR}" -maxdepth 1 -type f -name '*.whl' | sort)

if [[ "${#CORE_DEBS[@]}" -ne 1 || "${#EXTRAS_TARS[@]}" -ne 1 || "${#WHEELS[@]}" -ne 1 ]]; then
  echo "Expected exactly one NEAT core .deb, one extras tar.gz, and one wheel in ${ARTIFACTS_DIR}." >&2
  echo "core debs: ${#CORE_DEBS[@]}" >&2
  printf '  %s\n' "${CORE_DEBS[@]}" >&2 || true
  echo "extras tars: ${#EXTRAS_TARS[@]}" >&2
  printf '  %s\n' "${EXTRAS_TARS[@]}" >&2 || true
  echo "wheels: ${#WHEELS[@]}" >&2
  printf '  %s\n' "${WHEELS[@]}" >&2 || true
  exit 1
fi

INSTALL_SCRIPT_PATH="${ARTIFACTS_DIR}/${INSTALL_SCRIPT_NAME}"
[[ -f "${INSTALL_SCRIPT_PATH}" ]] || { echo "Missing install script: ${INSTALL_SCRIPT_PATH}" >&2; exit 1; }

for metadata_file in metadata.json metadata-minimal.json metadata-all.json; do
  if [[ ! -f "${ARTIFACTS_DIR}/${metadata_file}" ]]; then
    echo "Missing package metadata from build output: ${ARTIFACTS_DIR}/${metadata_file}" >&2
    echo "Run ./build.sh --all or ./build.sh --fuzz with a sima-cli that supports packages build." >&2
    exit 1
  fi
done

if ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "dpkg-deb is required to validate core DEB contents." >&2
  exit 1
fi

tmp_dir="$(mktemp -d /tmp/sima-neat-upload-XXXXXX)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

core_extract_dir="${tmp_dir}/core-deb-extract"
mkdir -p "${core_extract_dir}"
dpkg-deb -x "${CORE_DEBS[0]}" "${core_extract_dir}"
for required_cli in usr/bin/fix_devkit_runtime.sh usr/bin/neat; do
  if [[ ! -x "${core_extract_dir}/${required_cli}" ]]; then
    echo "Core DEB missing executable ${required_cli}." >&2
    exit 1
  fi
done

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
find "${ARTIFACTS_DIR}" -maxdepth 1 -type f \
  -exec cp -f {} "${OUTPUT_DIR}/" \;

echo "Prepared upload artifacts:"
ls -lh "${OUTPUT_DIR}"
