#!/usr/bin/env bash
set -euo pipefail

# Same direct-API fixtures as tests/tools/prepare_genai_models.sh; no compilation.
model_root="${SIMAPCIE_GENAI_MODELS_PATH:-${HOME}/workspace/models_genai}"
text_model="${SIMA_TEST_LLIMA_TEXT_MODEL:-Qwen2.5-0.5B-Instruct-Autoround-a16w4}"
vlm_model="${SIMA_TEST_LLIMA_VLM_MODEL:-LFM2.5-VL-450M-Autoround-a16w4}"
asr_model="${SIMA_TEST_LLIMA_ASR_MODEL:-whisper-small-a16w8}"
asr_repo="${SIMA_TEST_LLIMA_ASR_REPO:-simaai/whisper-small-a16w8}"

if [[ "${1:-}" == --help ]]; then
  echo "Download prepared LLM, VLM and Whisper fixtures to ${model_root}."
  echo "Override SIMAPCIE_GENAI_MODELS_PATH and SIMA_TEST_LLIMA_{TEXT,VLM,ASR}_MODEL."
  exit 0
fi
if (( $# != 0 )); then
  echo "ERROR: this script takes no arguments (use --help)." >&2
  exit 1
fi
command -v hf >/dev/null || { echo "ERROR: hf CLI is required." >&2; exit 1; }

download_model() {
  local name="$1" repo="$2" config="$3" revision="${4:-}"
  if [[ -z "${name}" || "${name}" == */* || "${name}" == *..* ]]; then
    echo "ERROR: model names must be directory names, not paths: ${name}" >&2
    exit 1
  fi
  local args=("${repo}" --local-dir "${model_root}/${name}")
  if [[ -n "${revision}" ]]; then
    args+=(--revision "${revision}")
  fi
  hf download "${args[@]}"
  test -f "${model_root}/${name}/${config}" || {
    echo "ERROR: missing ${config} in ${model_root}/${name}" >&2
    exit 1
  }
}

mkdir -p "${model_root}"
# Avoid concurrent jobs modifying a fixture while it is being downloaded.
exec 9>"${model_root}/.prepare-pcie-genai-models.lock"
flock --wait 1800 9
download_model "${text_model}" "simaai/${text_model}" devkit/vlm_config.json
download_model "${vlm_model}" "simaai/${vlm_model}" devkit/vlm_config.json
download_model "${asr_model}" "${asr_repo}" devkit/whisper_config.json \
  b34d5a300e74fcd1e9bc46bd70bd23cfc2b6b8d3
echo "[pcie-genai-models] ready under ${model_root}"
