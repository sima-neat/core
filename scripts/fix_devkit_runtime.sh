#!/usr/bin/env bash
set +e

# Recovery script to bring a devkit out of a bad runtime state by
# restarting remote processors and related services.
if [[ $# -gt 0 ]]; then
  pass="$1"
else
  pass="${DEVKIT_PASSWORD:-edgeai}"
fi
run_step() {
  local label="$1"
  shift
  printf "[recovery] %s...\n" "$label"
  printf '%s\n' "$pass" | sudo -S -p '' "$@"
  local rc=$?
  printf "[recovery] %s rc=%d\n" "$label" "$rc"
  return $rc
}

run_step "remoteproc0 stop" sh -c 'echo stop > /sys/class/remoteproc/remoteproc0/state'
run_step "remoteproc1 stop" sh -c 'echo stop > /sys/class/remoteproc/remoteproc1/state'
run_step "remoteproc1 start" sh -c 'echo start > /sys/class/remoteproc/remoteproc1/state'
run_step "remoteproc0 start" sh -c 'echo start > /sys/class/remoteproc/remoteproc0/state'
run_step "remoteproc status" sh -c 'for rp in /sys/class/remoteproc/remoteproc0 /sys/class/remoteproc/remoteproc1; do echo "$rp: $(cat $rp/name) state=$(cat $rp/state)"; done'
run_step "init_mla_memory" /usr/bin/init_mla_memory.sh
run_step "restart simaai-appcomplex.service" systemctl restart simaai-appcomplex.service
run_step "restart simaai-pipeline-manager.service" systemctl restart simaai-pipeline-manager.service
run_step "restart rctd.service" systemctl restart rctd.service
