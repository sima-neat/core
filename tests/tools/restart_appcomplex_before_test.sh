#!/usr/bin/env bash
set -euo pipefail

# TODO: Remove this workaround once CI runners provide MLA-RT with complete
# BlockAllocator mapping and IRQ-thread teardown.
echo "[genai-tests] restarting simaai-appcomplex.service for test isolation"
sudo -n systemctl restart simaai-appcomplex.service

for _ in {1..30}; do
  if systemctl is-active --quiet simaai-appcomplex.service; then
    daemon_pid="$(
      systemctl show \
        --property=MainPID \
        --value \
        simaai-appcomplex.service
    )"
    if [[ "${daemon_pid}" =~ ^[1-9][0-9]*$ ]]; then
      exit 0
    fi
  fi
  sleep 1
done

echo "ERROR: simaai-appcomplex.service did not become ready after restart." >&2
exit 1
