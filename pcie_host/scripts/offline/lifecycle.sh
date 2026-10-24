#!/usr/bin/env bash
# Adapt the published installers; retain producer admission and runtime lifecycle.
set -euo pipefail
role=$1
phase=$2
payload=$3
plan=$4
cd "$payload"
if [[ "$role" == host ]]; then
  if [[ "$phase" == preflight ]]; then
    python3 ./pcie_delivery.py install-list --directory "$payload" --extra-dependency python3-venv >/dev/null
  else
    bash ./install_pciehost.sh --skip-setup --python --python-venv /opt/sima/neat/pyneatpcie
  fi
else
  source ./install_neat_framework.sh
  parse_args --local
  collect_debs_in_install_order . DEBS
  ENV_MODE="$(detect_env_mode)"
  [[ "$ENV_MODE" == modalix-board ]] || { echo 'Expected a provisioned Modalix card' >&2; exit 1; }
  ensure_platform_compatible
  prepare_board_platform_admission
  verify_board_platform_plan "$plan"
  verify_simulated_package_removals "$plan" "${DEBS[@]}"
  verify_scheduler_transaction "$plan"
  if [[ "$phase" == install ]]; then
    # Sentinel is optional telemetry, not a Neat runtime dependency. Never invoke
    # its online updater. Existing Sentinel remains available to the runtime.
    install_board_sentinel() { log 'Offline install: retaining existing optional Sentinel telemetry'; }
    # deploy.py already installs the complete wheelhouse into an isolated venv.
    # The online installer would reopen it with unrelated system Python packages.
    install_python_environment() { log 'Offline install: retaining prepared isolated Python environment'; }
    install_for_environment
  fi
fi
