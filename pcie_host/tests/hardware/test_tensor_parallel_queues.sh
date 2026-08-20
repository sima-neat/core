#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${SIMAPCIE_YOLOV8_MODEL:-}"
QUEUES="${SIMAPCIE_STRESS_QUEUES:-0 1 2 3}"
ITERATIONS="${SIMAPCIE_STRESS_ITERATIONS:-1000}"
LOG_DIR="${SIMAPCIE_STRESS_LOG_DIR:-${TMPDIR:-/tmp}/sima-pcie-host-parallel-queues-$$}"

if [[ -z "${MODEL}" || ! -f "${MODEL}" ]]; then
  echo "ERROR: SIMAPCIE_YOLOV8_MODEL is not set to a readable model: ${MODEL}" >&2
  exit 1
fi
if [[ ! -x "${SCRIPT_DIR}/test_tensor_run" ]]; then
  echo "ERROR: missing executable: ${SCRIPT_DIR}/test_tensor_run" >&2
  exit 1
fi
if ! [[ "${ITERATIONS}" =~ ^[0-9]+$ ]] || [[ "${ITERATIONS}" -le 0 ]]; then
  echo "ERROR: SIMAPCIE_STRESS_ITERATIONS must be a positive integer." >&2
  exit 1
fi

rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"

echo "PCIe tensor parallel queue stress"
echo "  model=${MODEL}"
echo "  queues=${QUEUES}"
echo "  push_pull_iterations_per_queue=${ITERATIONS}"
echo "  log_dir=${LOG_DIR}"

run_queue() {
  local queue="$1"
  local log_file

  log_file="${LOG_DIR}/queue-${queue}.log"
  echo "queue ${queue}: ${ITERATIONS} push/pull iteration(s)"
  "${SCRIPT_DIR}/test_tensor_run" \
    --model "${MODEL}" \
    --queue "${queue}" \
    --iterations "${ITERATIONS}" \
    --card-gst-debug-file "/tmp/q${queue}-card.gst.log" \
    >"${log_file}" 2>&1
}

pids=()
queue_labels=()
for queue in ${QUEUES}; do
  run_queue "${queue}" &
  pids+=("$!")
  queue_labels+=("${queue}")
done

failed=0
for index in "${!pids[@]}"; do
  pid="${pids[${index}]}"
  queue="${queue_labels[${index}]}"
  if wait "${pid}"; then
    echo "queue ${queue}: passed"
  else
    echo "queue ${queue}: failed" >&2
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "Parallel queue stress failed. Recent logs:" >&2
  for log_file in "${LOG_DIR}"/*.log; do
    [[ -f "${log_file}" ]] || continue
    echo "===== ${log_file} =====" >&2
    tail -n 80 "${log_file}" >&2 || true
  done
  exit 1
fi

echo "parallel queue stress passed"
