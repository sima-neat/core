#!/usr/bin/env bash
# Check the SDK against the selected Scheduler's pinned sysroot and fetch the
# Runner's exact platform archives into a fresh folder. The sysroot is changed
# only when NEAT_UPDATE_SYSROOT=ON (build.sh --update-sysroot).
# stdout is exclusively shell exports; provisioning and validation use stderr.
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: prepare_scheduler_sdk.sh FRESH_ARCHIVE_CACHE' >&2; exit 2; }
package_dir=$(realpath -m "$1")
[[ ! -e "$package_dir" ]] || { echo 'SDK archive cache must be fresh' >&2; exit 1; }
mkdir -p "$package_dir"
policy=$(python3 tools/build_dependencies.py sdk-policy --core "$PWD")
sysroot_version=$(python3 tools/ci/scheduler_sdk_policy.py --policy-json "$policy" --field sysroot-version)
# SDK images may set SDK_PKG_LIST with whitespace and empty entries.
pkg_list=$(tr -d '[:space:]' <<< "${SDK_PKG_LIST:-}" | sed 's/,,*/,/g; s/^,//; s/,$//')
update_command="sudo env SDK_PKG_LIST=$pkg_list sysroot update $sysroot_version"
if [[ "${NEAT_UPDATE_SYSROOT:-OFF}" == ON ]]; then
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
  --sysroot /opt/toolchain/aarch64/modalix --compiler "$CXX" > "$package_dir/sdk-verification.json"
cat "$package_dir/sdk-verification.json" >&2

# Download the exact platform DEBs the selected Runner consumes, with a private
# APT configuration so neither the sysroot nor the dpkg state changes. -B keeps
# bytecode out of the receipted Scheduler tools folder.
platform_debs=$(python3 -B - "$NEAT_SCHEDULER_PACKAGE_DIR" "$NEAT_RUNNER_PACKAGE_DIR" <<'PY'
import sys
from pathlib import Path
scheduler, runner = map(Path, sys.argv[1:])
prefix = scheduler / 'usr' if (scheduler / 'usr').is_dir() else scheduler
sys.path.insert(0, str(prefix / 'share/SimaScheduler/tools'))
from client_dependencies import platform_provider_contracts, receipt
for provider in platform_provider_contracts(receipt(runner, 'SimaRunner')).values():
    deb = provider['debian_provider']
    print(f"{deb['Package']}:{deb['Architecture']}={deb['Version']}")
PY
)
if [[ "$sysroot_version" == *~pre* ]]; then
  repository=${SYSROOT_PRE_RELEASE_REPOSITORY:-https://debian.neat.sima.ai/pre-release}
else
  repository=${SYSROOT_DAILY_REPOSITORY:-https://debian.neat.sima.ai/daily}
fi
apt_dir="$package_dir/apt"
mkdir -p "$apt_dir/lists/partial" "$apt_dir/empty" "$package_dir/downloads"
echo "deb [arch=arm64 trusted=yes] $repository agate non-free" > "$apt_dir/sources.list"
cat > "$apt_dir/apt.conf" <<EOF
Dir::Etc::sourcelist "$apt_dir/sources.list";
Dir::Etc::sourceparts "$apt_dir/empty";
Dir::Etc::preferencesparts "$apt_dir/empty";
Dir::State::lists "$apt_dir/lists";
Dir::State::status "/dev/null";
Dir::Cache::pkgcache "";
Dir::Cache::srcpkgcache "";
APT::Architecture "arm64";
EOF
if [[ -n "$platform_debs" ]]; then
  APT_CONFIG="$apt_dir/apt.conf" apt-get update -q >&2
  # shellcheck disable=SC2086 # one package=version per word
  (cd "$package_dir/downloads" && APT_CONFIG="$apt_dir/apt.conf" apt-get download $platform_debs >&2)
fi
# The coordinator checks each archive against the Runner receipt's hashes.
python3 tools/build_dependencies.py platform-archives --core "$PWD" \
  --platform-package-dir "$package_dir/downloads"
