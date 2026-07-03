#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
sanitize_path

mkdir -p "${ARTIFACT_DIR}/host" "${ARTIFACT_DIR}/card"

run_host() {
  local name="$1"
  shift
  {
    echo "+ $*"
    "$@"
  } >"${ARTIFACT_DIR}/host/${name}" 2>&1 || true
}

run_card() {
  local name="$1"
  shift
  {
    echo "+ $*"
    ssh_card "$@"
  } >"${ARTIFACT_DIR}/card/${name}" 2>&1 || true
}

run_host "gst-inspect-neatpciehost.txt" /usr/bin/gst-inspect-1.0 neatpciehost
run_host "dpkg-pciehost.txt" bash -lc "dpkg -l | grep -E 'sima-pcie-host|sima-neat' || true"
run_host "ldd-libgstneatpciehost.txt" bash -lc \
  "find /usr/lib /usr/local/lib -name libgstneatpciehost.so -print -quit 2>/dev/null | xargs -r ldd"

mapfile -t last_logs < <(
  find "${WORKSPACE}/_work" -type f -path '*/Testing/Temporary/LastTest.log' -print 2>/dev/null | sort
)
if [[ "${#last_logs[@]}" -gt 0 ]]; then
  cp "${last_logs[0]}" "${ARTIFACT_DIR}/host/LastTest.log" || true
fi

run_card "gst-inspect-neatpciesrc.txt" gst-inspect-1.0 neatpciesrc
run_card "gst-inspect-neatpciesink.txt" gst-inspect-1.0 neatpciesink
run_card "pcie-pipeline-builder-version.txt" bash -lc "command -v pcie-pipeline-builder; pcie-pipeline-builder --help | head -n 80"

queues="${QUEUE} ${STRESS_QUEUES}"
for queue in ${queues}; do
  [[ -n "${queue}" ]] || continue
  run_card "q${queue}.status.txt" cat "/run/sima-neat/pcie/q${queue}.status"
  run_card "q${queue}.log.txt" cat "/var/log/sima-neat/pcie/q${queue}.log"
  run_card "q${queue}-card.gst.log.txt" cat "/tmp/q${queue}-card.gst.log"
done

find "${ARTIFACT_DIR}" -type f -empty -delete 2>/dev/null || true
echo "Collected PCIe host hardware artifacts under ${ARTIFACT_DIR}"
