#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <model-name> <default-mpk> <executable> [args...]" >&2
  exit 2
fi

MODEL_NAME="$1"
DEFAULT_MPK="$2"
EXE="$3"
shift 3

# The C++ tutorials read the same variables as tests/tutorials/test_tutorials_run.py.
case "${MODEL_NAME}" in
  resnet_50) MODEL_ENV="SIMA_NEAT_TUTORIAL_MODEL_RESNET" ;;
  yolo_v8s) MODEL_ENV="SIMA_NEAT_TUTORIAL_MODEL_YOLO" ;;
  *)
    echo "$0: unknown tutorial model ${MODEL_NAME}" >&2
    exit 2
    ;;
esac

MPK="${!MODEL_ENV:-${DEFAULT_MPK}}"

# Fail here rather than in the tutorial: some tutorials skip a model path that does not exist.
if [[ ! -f "${MPK}" || ! -s "${MPK}" ]]; then
  echo "${MODEL_NAME} MPK is missing, empty or not a regular file: ${MPK}" >&2
  echo "Set ${MODEL_ENV} to an MPK, or run build.sh --download-models-only --model-target-folder ${DEFAULT_MPK%/*} to write ${DEFAULT_MPK}." >&2
  exit 1
fi

exec "${EXE}" --model "${MPK}" "$@"
