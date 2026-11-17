#!/usr/bin/env bash
# Check the active SDK against the selected Scheduler platform identity.
# Store the observed identity without downloading platform packages. The sysroot is changed
# only when NEAT_UPDATE_SYSROOT=ON (build.sh --update-sysroot).
# stdout is exclusively shell exports; provisioning and validation use stderr.
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: prepare_scheduler_sdk.sh FRESH_VERIFICATION_DIRECTORY' >&2; exit 2; }
verification_dir=$(realpath -m "$1")
[[ ! -e "$verification_dir" ]] || { echo 'SDK verification directory must be fresh' >&2; exit 1; }
mkdir -p "$verification_dir"
policy=$(python3 tools/build_dependencies.py sdk-policy --core "$PWD")
sysroot_version=$(python3 tools/ci/scheduler_sdk_policy.py --policy-json "$policy" --field sysroot-version)
# SDK images may set SDK_PKG_LIST with whitespace and empty entries.
pkg_list=$(tr -d '[:space:]' <<< "${SDK_PKG_LIST:-}" | sed 's/,,*/,/g; s/^,//; s/,$//')
update_command="sudo env SDK_PKG_LIST=$pkg_list sysroot update $sysroot_version"
if [[ "$sysroot_version" != *~* ]]; then
  # Official releases must already be present in the SDK cache, even when the
  # caller requested --update-sysroot. Never route these through daily overlays.
  python3 tools/ci/scheduler_sdk_policy.py --policy-json "$policy" --check-release-sdk >&2
elif [[ "${NEAT_UPDATE_SYSROOT:-OFF}" == ON ]]; then
  setup_env=("SDK_PKG_LIST=$pkg_list" "SYSROOT=/opt/toolchain/aarch64/modalix")
  if [[ $(id -u) == 0 ]]; then
    env "${setup_env[@]}" sysroot update "$sysroot_version" >&2
  else
    sudo env "${setup_env[@]}" sysroot update "$sysroot_version" >&2
  fi
fi
sysroot_status=$(sysroot status)
echo "$sysroot_status" >&2
# Without an active overlay, the sysroot is the one the SDK image shipped.
if [[ $(sed -n 's/^Sysroot overlay state: *//p' <<< "$sysroot_status") == inactive ]]; then
  active_version=$(sed -n 's/^SDK image platform: *//p' <<< "$sysroot_status")
else
  active_version=$(sed -n 's/^Sysroot overlay revision: *//p' <<< "$sysroot_status")
fi
if [[ "$active_version" != "$sysroot_version" ]]; then
  if [[ "${NEAT_UPDATE_SYSROOT:-OFF}" == ON ]]; then
    echo "ERROR: sysroot is at ${active_version:-unknown} after updating to $sysroot_version" >&2
    exit 1
  fi
  echo "WARNING: sysroot is at ${active_version:-unknown}, but the Scheduler set was built for $sysroot_version." >&2
  echo "WARNING: To match it, run: $update_command" >&2
fi
# The official environment selects the SDK compiler, not the host's default c++.
source /opt/bin/simaai-init-build-env modalix >&2
python3 tools/build_dependencies.py verify-sdk --core "$PWD" \
  --sysroot /opt/toolchain/aarch64/modalix --compiler "$CXX" > "$verification_dir/sdk-verification.json"
# Reject old coordinator output before dependency preparation can ask for the
# retired platform archive selection.
python3 - "$verification_dir/sdk-verification.json" <<'PY'
import json
import sys
from pathlib import Path

verification = json.loads(Path(sys.argv[1]).read_text())
platform = verification.get('build_platform') if isinstance(verification, dict) else None
if (not isinstance(platform, dict) or set(platform) != {'elxr_version', 'sysroot_version', 'target'}
        or any(not isinstance(value, str) or not value.strip() for value in platform.values())):
    raise SystemExit('ERROR: Scheduler SDK verification lacks build_platform; '
                     'rebuild Runner, then Scheduler, before building Core.')
PY
cat "$verification_dir/sdk-verification.json" >&2

# Only the selected SDK supplies platform libraries and package identity.
# Consumers receive the verified observations; no live APT lookup is required.
printf 'export NEAT_SDK_VERIFICATION_JSON=%q\n' "$verification_dir/sdk-verification.json"
