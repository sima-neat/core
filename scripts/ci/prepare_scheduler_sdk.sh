#!/usr/bin/env bash
# Prepare the disposable SDK from the selected Scheduler's receipt-pinned policy.
# stdout is exclusively shell exports; provisioning and validation use stderr.
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: prepare_scheduler_sdk.sh FRESH_ARCHIVE_CACHE' >&2; exit 2; }
package_dir=$(realpath -m "$1")
[[ ! -e "$package_dir" ]] || { echo 'SDK archive cache must be fresh' >&2; exit 1; }
mkdir -p "$package_dir"
policy=$(python3 tools/build_dependencies.py sdk-policy --core "$PWD")
sysroot_version=$(python3 - "$policy" <<'PY_POLICY'
import json, re, sys
from pathlib import Path
policy = json.loads(sys.argv[1])
expected = json.loads(Path('deps/manifest.json').read_text())['elxr_version']
version = policy.get('sysroot-version', '')
if policy.get('elxr_version') != expected:
    raise SystemExit('Selected Scheduler firmware differs from Core; rebuild/publish the matching Scheduler first')
base, channel, build = re.fullmatch(r'(.+)_(master|develop)_B([1-9][0-9]*)', expected).groups()
suffix = '~pre' + build if channel == 'master' else r'~git[0-9]{12}\.[0-9a-f]{7,40}-' + build
if not re.fullmatch(re.escape(base) + suffix, version):
    raise SystemExit('Selected Scheduler exact sysroot receipt differs from Core firmware')
print(version)
PY_POLICY
)
setup_env=("SDK_PKG_LIST=${SDK_PKG_LIST:-}" "SYSROOT=/opt/toolchain/aarch64/modalix"
  "SYSROOT_UPDATE_DOWNLOAD_DIR=$package_dir/downloads")
if [[ $(id -u) == 0 ]]; then
  env "${setup_env[@]}" sysroot update "$sysroot_version" >&2
else
  sudo env "${setup_env[@]}" sysroot update "$sysroot_version" >&2
fi
sysroot status >&2
# The official environment selects the SDK compiler, not the host's default c++.
source /opt/bin/simaai-init-build-env modalix >&2
python3 tools/build_dependencies.py verify-sdk --core "$PWD" \
  --sysroot /opt/toolchain/aarch64/modalix --compiler "$CXX" > "$package_dir/sdk-verification.json"
cat "$package_dir/sdk-verification.json" >&2
# Select the exact consumed Runner providers from this transaction's private
# download cache, then export only the retained verified archives.
archive_exports=$(python3 tools/build_dependencies.py platform-archives --core "$PWD" \
  --platform-package-dir "$package_dir/downloads")
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
