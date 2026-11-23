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

MPK="${!MODEL_ENV:-}"
FROM_ENV=1
if [[ -z "${MPK}" ]]; then
  MPK="${DEFAULT_MPK}"
  FROM_ENV=0
fi

# Fail here rather than in the tutorial: some tutorials skip a model path that does not exist.
if [[ ! -f "${MPK}" || ! -s "${MPK}" ]]; then
  echo "${MODEL_NAME} MPK is missing, empty or not a regular file: ${MPK}" >&2
  if [[ "${FROM_ENV}" == 1 ]]; then
    # Preparing the default file would not help: the variable still wins.
    echo "${MODEL_ENV} selects this path. Correct ${MODEL_ENV}, or unset it to use ${DEFAULT_MPK}." >&2
  else
    echo "Prepare ${DEFAULT_MPK} as described in Tutorial Setup, Prepare model archives, or set ${MODEL_ENV} to an MPK." >&2
  fi
  exit 1
fi

exec "${EXE}" --model "${MPK}" "$@"
