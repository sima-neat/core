#!/usr/bin/env bash
# Prepare the disposable SDK from the selected Scheduler's receipt-pinned policy.
# stdout is exclusively shell exports; provisioning and validation use stderr.
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: prepare_scheduler_sdk.sh FRESH_ARCHIVE_CACHE' >&2; exit 2; }
package_dir=$(realpath -m "$1")
[[ ! -e "$package_dir" ]] || { echo 'SDK archive cache must be fresh' >&2; exit 1; }
mkdir -p "$package_dir"
policy=$(python3 tools/build_dependencies.py sdk-policy --core "$PWD")
sysroot_version=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["sysroot-version"])' "$policy")
case "$sysroot_version" in
  *~git*) channel=daily; revision="";;
  *~pre*) channel=pre-release; revision="${sysroot_version##*~pre}";;
  *) echo "Unsupported committed Scheduler sysroot version: $sysroot_version" >&2; exit 2;;
esac
setup_env=("SDK_APT_CHANNEL=$channel" "SDK_LINUX_LIBC_VERSION=${SDK_LINUX_LIBC_VERSION:-}"
  "SIMAAI_PLATFORM_BUILD_REVISION=$revision" "SIMAAI_VALIDATE_TARGET_ORIGIN=1"
  "SYSROOT=/opt/toolchain/aarch64/modalix")
setup=(/usr/local/bin/setup-sdk-sysroot.sh "$sysroot_version" "${SDK_PKG_LIST:-}")
if [[ $(id -u) == 0 ]]; then
  env "${setup_env[@]}" "${setup[@]}" >&2
else
  sudo env "${setup_env[@]}" "${setup[@]}" >&2
fi
# The official environment selects the SDK compiler, not the host's default c++.
source /opt/bin/simaai-init-build-env modalix >&2
python3 tools/build_dependencies.py verify-sdk --core "$PWD" \
  --sysroot /opt/toolchain/aarch64/modalix --compiler "$CXX" > "$package_dir/sdk-verification.json"
cat "$package_dir/sdk-verification.json" >&2
# The official helper retains its downloads at /tmp/modalix. Select the exact
# consumed Runner providers there, then retain only those verified archives.
archive_exports=$(python3 tools/build_dependencies.py platform-archives --core "$PWD" \
  --platform-package-dir /tmp/modalix)
eval "$archive_exports"
python3 - "$package_dir" <<'PY_COPY'
import hashlib, json, os, shutil, sys
from pathlib import Path
out = Path(sys.argv[1])
for value in json.loads(os.environ['NEAT_RUNNER_PLATFORM_RUNTIME_DEBS']):
    source = Path(value)
    if source.is_symlink() or not source.is_file():
        raise SystemExit('Invalid selected SDK archive: ' + str(source))
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    target = out / source.name
    if target.exists():
        raise SystemExit('Duplicate selected SDK archive: ' + source.name)
    shutil.copyfile(source, target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise SystemExit('SDK archive changed during retention')
PY_COPY
# Recheck controls and receipt hashes after copying; only retained paths escape.
python3 tools/build_dependencies.py platform-archives --core "$PWD" \
  --platform-package-dir "$package_dir"
