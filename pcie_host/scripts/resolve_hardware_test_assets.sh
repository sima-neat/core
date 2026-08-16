#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: resolve_hardware_test_assets.sh [options]

Resolve model and image paths for PCIe host hardware tests. Explicit
environment variables win, local caches are searched next, and downloads are
used only when configured or modelzoo can provide the model.

Options:
  --workspace <dir>       Workspace/repo root to search (default: current dir)
  --extras-root <dir>     Extracted pciehost extras root for bundled test image
  --cache-dir <dir>       Download/cache directory (default: _work/pciehost-assets)
  --env-file <path>       Write KEY=VALUE lines for GitHub env/source usage
  -h, --help              Show this help

Inputs:
  SIMAPCIE_YOLOV8_MODEL       Existing YOLOv8 model tar.gz path
  SIMAPCIE_TEST_IMAGE         Existing test image path
  SIMAPCIE_BOXDECODE_IMAGE    Existing object-detection image path
  SIMAPCIE_YOLOV8_MODEL_URL   Direct URL fallback for YOLOv8 model
  SIMAPCIE_TEST_IMAGE_URL     Direct URL fallback for test image
  SIMAPCIE_BOXDECODE_IMAGE_URL
                              Direct URL fallback for object-detection image
  SIMAPCIE_YOLOV8_MODEL_NAME  modelzoo name (default: yolo_v8s)

Outputs:
  SIMAPCIE_YOLOV8_MODEL=<path>
  SIMAPCIE_TEST_IMAGE=<path>
  SIMAPCIE_BOXDECODE_IMAGE=<path>
EOF
}

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
EXTRAS_ROOT=""
CACHE_DIR=""
ENV_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      WORKSPACE="${2:-}"
      [[ -n "${WORKSPACE}" ]] || { echo "ERROR: --workspace requires a value" >&2; exit 1; }
      shift 2
      ;;
    --extras-root)
      EXTRAS_ROOT="${2:-}"
      [[ -n "${EXTRAS_ROOT}" ]] || { echo "ERROR: --extras-root requires a value" >&2; exit 1; }
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="${2:-}"
      [[ -n "${CACHE_DIR}" ]] || { echo "ERROR: --cache-dir requires a value" >&2; exit 1; }
      shift 2
      ;;
    --env-file)
      ENV_FILE="${2:-}"
      [[ -n "${ENV_FILE}" ]] || { echo "ERROR: --env-file requires a value" >&2; exit 1; }
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

WORKSPACE="$(cd "${WORKSPACE}" && pwd)"
if [[ -z "${CACHE_DIR}" ]]; then
  CACHE_DIR="${WORKSPACE}/_work/pciehost-assets"
fi
mkdir -p "${CACHE_DIR}"
CACHE_DIR="$(cd "${CACHE_DIR}" && pwd)"

if [[ -n "${EXTRAS_ROOT}" ]]; then
  EXTRAS_ROOT="$(cd "${EXTRAS_ROOT}" && pwd)"
fi

is_file() {
  [[ -n "${1:-}" && -f "$1" ]]
}

absolute_path() {
  local path="$1"
  local dir base
  dir="$(cd "$(dirname "${path}")" && pwd)"
  base="$(basename "${path}")"
  printf '%s/%s\n' "${dir}" "${base}"
}

first_existing_file() {
  local path
  for path in "$@"; do
    if is_file "${path}"; then
      absolute_path "${path}"
      return 0
    fi
  done
  return 1
}

find_cached_model() {
  local pattern="$1"
  find \
    "${WORKSPACE}" \
    "${CACHE_DIR}" \
    "${HOME}/.simaai/modelzoo" \
    "${HOME}/.sima/modelzoo" \
    "/data/simaai/modelzoo" \
    -type f -name "${pattern}" -print 2>/dev/null \
    | sort \
    | head -n 1
}

download_file() {
  local url="$1"
  local dest="$2"
  if [[ -z "${url}" ]]; then
    return 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required to download ${url}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${dest}")"
  curl -fsSL "${url}" -o "${dest}"
}

resolve_from_modelzoo() {
  local model_name="$1"
  local pattern="$2"
  [[ -n "${model_name}" ]] || return 1
  if command -v sima-cli >/dev/null 2>&1; then
    SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli modelzoo get "${model_name}" >/dev/null
  else
    return 1
  fi
  find_cached_model "${pattern}"
}

resolve_yolov8_model() {
  local value="${SIMAPCIE_YOLOV8_MODEL:-${SIMA_MODEL_TAR:-${SIMA_YOLO_TAR:-}}}"
  if is_file "${value}"; then
    absolute_path "${value}"
    return
  fi
  if first_existing_file \
      "${WORKSPACE}/models/yolo_v8s.tar.gz" \
      "${WORKSPACE}/models/yolov8s.tar.gz" \
      "${WORKSPACE}/../models/yolo_v8s.tar.gz" \
      "${WORKSPACE}/../models/yolov8s.tar.gz"; then
    return
  fi
  local found
  found="$(find_cached_model '*yolo*v8*s*.tar.gz' || true)"
  if is_file "${found}"; then
    absolute_path "${found}"
    return
  fi
  if [[ -n "${SIMAPCIE_YOLOV8_MODEL_URL:-}" ]]; then
    local dest="${CACHE_DIR}/$(basename "${SIMAPCIE_YOLOV8_MODEL_URL%%\?*}")"
    download_file "${SIMAPCIE_YOLOV8_MODEL_URL}" "${dest}"
    absolute_path "${dest}"
    return
  fi
  found="$(resolve_from_modelzoo "${SIMAPCIE_YOLOV8_MODEL_NAME:-yolo_v8s}" '*yolo*v8*s*.tar.gz' || true)"
  if is_file "${found}"; then
    absolute_path "${found}"
    return
  fi
  if first_existing_file \
      "${WORKSPACE}/models/yolov8n_mod_1_inputs_mpk_mlatess_bf16.tar.gz" \
      "${WORKSPACE}/../models/yolov8n_mod_1_inputs_mpk_mlatess_bf16.tar.gz"; then
    return
  fi
  echo "ERROR: failed to resolve YOLOv8 model. Set SIMAPCIE_YOLOV8_MODEL or SIMAPCIE_YOLOV8_MODEL_URL." >&2
  exit 1
}

resolve_test_image() {
  local value="${SIMAPCIE_TEST_IMAGE:-}"
  if is_file "${value}"; then
    absolute_path "${value}"
    return
  fi
  if [[ -n "${EXTRAS_ROOT}" ]] &&
     first_existing_file "${EXTRAS_ROOT}/share/sima-pcie-host/test-assets/test_image.jpg"; then
    return
  fi
  if first_existing_file \
      "${WORKSPACE}/pcie_host/tests/hardware/assets/test_image.jpg" \
      "${WORKSPACE}/core/pcie_host/tests/hardware/assets/test_image.jpg" \
      "${WORKSPACE}/tests/test_image_run/test_image.jpg" \
      "${WORKSPACE}/tests/test_image_boxdecode_run/test_image.jpg" \
      "${WORKSPACE}/../tests/test_image_run/test_image.jpg" \
      "${WORKSPACE}/../tests/test_image_boxdecode_run/test_image.jpg"; then
    return
  fi
  local url="${SIMAPCIE_TEST_IMAGE_URL:-https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg}"
  local dest="${CACHE_DIR}/test_image.jpg"
  download_file "${url}" "${dest}"
  absolute_path "${dest}"
}

resolve_boxdecode_image() {
  local value="${SIMAPCIE_BOXDECODE_IMAGE:-}"
  if is_file "${value}"; then
    absolute_path "${value}"
    return
  fi
  if first_existing_file \
      "${WORKSPACE}/tests/images/people.jpg" \
      "${WORKSPACE}/core/tests/images/people.jpg" \
      "${WORKSPACE}/../tests/images/people.jpg"; then
    return
  fi
  local url="${SIMAPCIE_BOXDECODE_IMAGE_URL:-https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg}"
  local dest="${CACHE_DIR}/boxdecode_image.jpg"
  download_file "${url}" "${dest}"
  absolute_path "${dest}"
}

YOLOV8_MODEL="$(resolve_yolov8_model)"
TEST_IMAGE="$(resolve_test_image)"
BOXDECODE_IMAGE="$(resolve_boxdecode_image)"

{
  printf 'SIMAPCIE_YOLOV8_MODEL=%s\n' "${YOLOV8_MODEL}"
  printf 'SIMAPCIE_TEST_IMAGE=%s\n' "${TEST_IMAGE}"
  printf 'SIMAPCIE_BOXDECODE_IMAGE=%s\n' "${BOXDECODE_IMAGE}"
} | tee "${ENV_FILE:-/dev/stdout}" >/dev/null

printf 'Resolved PCIe host hardware test assets:\n'
printf '  SIMAPCIE_YOLOV8_MODEL=%s\n' "${YOLOV8_MODEL}"
printf '  SIMAPCIE_TEST_IMAGE=%s\n' "${TEST_IMAGE}"
printf '  SIMAPCIE_BOXDECODE_IMAGE=%s\n' "${BOXDECODE_IMAGE}"
