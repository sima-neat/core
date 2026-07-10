#!/usr/bin/env bash
set -euo pipefail

DEFAULT_LLIMA_MODELS_PATH="/media/nvme/llima/models"
DEFAULT_TEXT_MODEL="Qwen2.5-0.5B-Instruct-GPTQ-a16w4"
DEFAULT_VLM_MODEL="LFM2.5-VL-450M-a16w4"
DEFAULT_ASR_MODEL="whisper-small-a16w8"

usage() {
  cat <<EOF
Usage: $(basename "$0")

Downloads the GenAI model fixtures used by NEAT tests.

Environment:
  LLIMA_MODELS_PATH              Model root. Default: ${DEFAULT_LLIMA_MODELS_PATH}
  SIMA_TEST_LLIMA_TEXT_MODEL     Text model name. Default: ${DEFAULT_TEXT_MODEL}
  SIMA_TEST_LLIMA_VLM_MODEL      VLM model name. Default: ${DEFAULT_VLM_MODEL}
  SIMA_TEST_LLIMA_ASR_MODEL      ASR model name. Default: ${DEFAULT_ASR_MODEL}
  SIMA_TEST_GENAI_FORCE_DOWNLOAD Set to 1 to run hf download even when the expected config exists.

The SIMA_TEST_LLIMA_*_MODEL values must be model directory names under
LLIMA_MODELS_PATH, not absolute paths and not Hugging Face repo ids.
EOF
}

validate_model_name() {
  local env_name="$1"
  local model_name="$2"

  if [[ -z "${model_name}" ]]; then
    echo "ERROR: ${env_name} must not be empty." >&2
    exit 1
  fi
  if [[ "${model_name}" = /* || "${model_name}" == *"/"* || "${model_name}" == *".."* ]]; then
    echo "ERROR: ${env_name} must be a model directory name under LLIMA_MODELS_PATH: ${model_name}" >&2
    echo "       Set LLIMA_MODELS_PATH for the root path and omit the simaai/ prefix." >&2
    exit 1
  fi
}

download_model() {
  local label="$1"
  local model_name="$2"
  local expected_config="$3"
  local target_dir="${LLIMA_MODELS_PATH}/${model_name}"

  if [[ "${SIMA_TEST_GENAI_FORCE_DOWNLOAD:-0}" != "1" && -f "${target_dir}/${expected_config}" ]]; then
    echo "[genai-models] ${label}: using existing ${target_dir}"
    return
  fi

  mkdir -p "${target_dir}"
  echo "[genai-models] ${label}: downloading simaai/${model_name} to ${target_dir}"
  hf download "simaai/${model_name}" --local-dir "${target_dir}"

  if [[ ! -f "${target_dir}/${expected_config}" ]]; then
    echo "ERROR: downloaded ${label} model is missing ${expected_config}: ${target_dir}" >&2
    exit 1
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if (( $# > 0 )); then
  echo "ERROR: unsupported argument: $1" >&2
  usage >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: hf CLI is required. Install sima-cli or huggingface_hub before running this script." >&2
  exit 1
fi

LLIMA_MODELS_PATH="${LLIMA_MODELS_PATH:-${DEFAULT_LLIMA_MODELS_PATH}}"
SIMA_TEST_LLIMA_TEXT_MODEL="${SIMA_TEST_LLIMA_TEXT_MODEL:-${DEFAULT_TEXT_MODEL}}"
SIMA_TEST_LLIMA_VLM_MODEL="${SIMA_TEST_LLIMA_VLM_MODEL:-${DEFAULT_VLM_MODEL}}"
SIMA_TEST_LLIMA_ASR_MODEL="${SIMA_TEST_LLIMA_ASR_MODEL:-${DEFAULT_ASR_MODEL}}"

validate_model_name "SIMA_TEST_LLIMA_TEXT_MODEL" "${SIMA_TEST_LLIMA_TEXT_MODEL}"
validate_model_name "SIMA_TEST_LLIMA_VLM_MODEL" "${SIMA_TEST_LLIMA_VLM_MODEL}"
validate_model_name "SIMA_TEST_LLIMA_ASR_MODEL" "${SIMA_TEST_LLIMA_ASR_MODEL}"

mkdir -p "${LLIMA_MODELS_PATH}"

download_model "text" "${SIMA_TEST_LLIMA_TEXT_MODEL}" "devkit/vlm_config.json"
download_model "vlm" "${SIMA_TEST_LLIMA_VLM_MODEL}" "devkit/vlm_config.json"
download_model "asr" "${SIMA_TEST_LLIMA_ASR_MODEL}" "devkit/whisper_config.json"

echo "[genai-models] ready under ${LLIMA_MODELS_PATH}"
