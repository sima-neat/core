#!/usr/bin/env bash
set -euo pipefail

DEFAULT_LLIMA_MODELS_PATH="/media/nvme/llima/models"
DEFAULT_TEXT_MODEL="Qwen2.5-0.5B-Instruct-GPTQ-a16w4"
DEFAULT_VLM_MODEL="LFM2.5-VL-450M-a16w4"
DEFAULT_ASR_MODEL="whisper-small-a16w8"
DEFAULT_ASR_REPO="florianvoss/whisper-small-a16w8"
DEFAULT_REASONING_QWEN_MODEL="Qwen3-0.6B-GPTQ-a16w4"
DEFAULT_REASONING_QWEN_REPO="simaai/Qwen3-0.6B-GPTQ-a16w4"
DEFAULT_REASONING_QWEN_REVISION="964b272f2b62a434b10b0c14f605016fd285ca5e"
DEFAULT_REASONING_GEMMA_MODEL="Gemma-4-E2B-it-TextOnly-GPTQ-a16w4"
DEFAULT_REASONING_GEMMA_REPO="florianvoss/Gemma-4-E2B-it-TextOnly-GPTQ-a16w4"
DEFAULT_REASONING_GEMMA_REVISION="1c72307f46ad055a641420310b36a061715b0508"
usage() {
  cat <<EOF
Usage: $(basename "$0")

Downloads the GenAI model fixtures used by NEAT tests.

Environment:
  LLIMA_MODELS_PATH              Model root. Default: ${DEFAULT_LLIMA_MODELS_PATH}
  SIMA_TEST_LLIMA_TEXT_MODEL     Text model name. Default: ${DEFAULT_TEXT_MODEL}
  SIMA_TEST_LLIMA_VLM_MODEL      VLM model name. Default: ${DEFAULT_VLM_MODEL}
  SIMA_TEST_LLIMA_ASR_MODEL      ASR model name. Default: ${DEFAULT_ASR_MODEL}
  SIMA_TEST_LLIMA_ASR_REPO       ASR Hugging Face repo. Default: ${DEFAULT_ASR_REPO}
  SIMA_TEST_LLIMA_REASONING_QWEN_MODEL
                                  Qwen reasoning model. Default: ${DEFAULT_REASONING_QWEN_MODEL}
  SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL
                                  Gemma reasoning model. Default: ${DEFAULT_REASONING_GEMMA_MODEL}

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
  local repo_id="$2"
  local model_name="$3"
  local expected_config="$4"
  local revision="${5:-}"
  local target_dir="${LLIMA_MODELS_PATH}/${model_name}"

  mkdir -p "${target_dir}"
  echo "[genai-models] ${label}: synchronizing ${repo_id} to ${target_dir}"
  local args=("${repo_id}" --local-dir "${target_dir}")
  if [[ -n "${revision}" ]]; then
    args+=(--revision "${revision}")
  fi
  hf download "${args[@]}"

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
SIMA_TEST_LLIMA_ASR_REPO="${SIMA_TEST_LLIMA_ASR_REPO:-${DEFAULT_ASR_REPO}}"
SIMA_TEST_LLIMA_REASONING_QWEN_MODEL="${SIMA_TEST_LLIMA_REASONING_QWEN_MODEL:-${DEFAULT_REASONING_QWEN_MODEL}}"
SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL="${SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL:-${DEFAULT_REASONING_GEMMA_MODEL}}"

validate_model_name "SIMA_TEST_LLIMA_TEXT_MODEL" "${SIMA_TEST_LLIMA_TEXT_MODEL}"
validate_model_name "SIMA_TEST_LLIMA_VLM_MODEL" "${SIMA_TEST_LLIMA_VLM_MODEL}"
validate_model_name "SIMA_TEST_LLIMA_ASR_MODEL" "${SIMA_TEST_LLIMA_ASR_MODEL}"
validate_model_name "SIMA_TEST_LLIMA_REASONING_QWEN_MODEL" "${SIMA_TEST_LLIMA_REASONING_QWEN_MODEL}"
validate_model_name "SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL" "${SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL}"

mkdir -p "${LLIMA_MODELS_PATH}"

download_model \
  "text" "simaai/${SIMA_TEST_LLIMA_TEXT_MODEL}" "${SIMA_TEST_LLIMA_TEXT_MODEL}" \
  "devkit/vlm_config.json"
download_model \
  "vlm" "simaai/${SIMA_TEST_LLIMA_VLM_MODEL}" "${SIMA_TEST_LLIMA_VLM_MODEL}" \
  "devkit/vlm_config.json"
download_model \
  "asr" "${SIMA_TEST_LLIMA_ASR_REPO}" "${SIMA_TEST_LLIMA_ASR_MODEL}" \
  "devkit/whisper_config.json"
download_model \
  "Qwen reasoning" "${DEFAULT_REASONING_QWEN_REPO}" \
  "${SIMA_TEST_LLIMA_REASONING_QWEN_MODEL}" "devkit/vlm_config.json" \
  "${DEFAULT_REASONING_QWEN_REVISION}"
download_model \
  "Gemma reasoning" "${DEFAULT_REASONING_GEMMA_REPO}" \
  "${SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL}" "devkit/vlm_config.json" \
  "${DEFAULT_REASONING_GEMMA_REVISION}"

echo "[genai-models] ready under ${LLIMA_MODELS_PATH}"
