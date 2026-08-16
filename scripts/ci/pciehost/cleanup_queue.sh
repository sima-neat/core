#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

cleanup_host_pcie_device

ssh_card "REMOTE_QUEUE=$(shell_quote "${QUEUE}") REMOTE_STRESS_QUEUES=$(shell_quote "${STRESS_QUEUES}") bash -s" <<'REMOTE_CLEANUP' || true
set -euo pipefail

queues="${REMOTE_QUEUE} ${REMOTE_STRESS_QUEUES}"
for queue in ${queues}; do
  [[ -n "${queue}" ]] || continue
  pattern="[p]cie-pipeline-builder.*--queue([ =])?${queue}([^0-9]|$)"
  if command -v pgrep >/dev/null 2>&1; then
    mapfile -t pids < <(pgrep -f "${pattern}" || true)
    if [[ "${#pids[@]}" -gt 0 ]]; then
      kill "${pids[@]}" 2>/dev/null || true
      sleep 1
      mapfile -t pids < <(pgrep -f "${pattern}" || true)
      if [[ "${#pids[@]}" -gt 0 ]]; then
        kill -9 "${pids[@]}" 2>/dev/null || true
      fi
    fi
  fi

  rm -f \
    "/run/sima-neat/pcie/q${queue}.pid" \
    "/run/sima-neat/pcie/q${queue}.status" \
    "/tmp/q${queue}-card.gst.log" \
    2>/dev/null || true
done
REMOTE_CLEANUP
