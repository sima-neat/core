"""Focused tests for native Modalix package handling in the installer."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "tools" / "install_neat_framework.sh"
RECOVERY = ROOT / "scripts" / "fix_devkit_runtime.sh"


def run_bash(
    script: str, target: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Resolve INSTALLER at call time so --installer/--recovery still apply."""
    return subprocess.run(
        ["bash", "-c", script, "bash", str(target or INSTALLER)],
        check=False,
        text=True,
        capture_output=True,
    )


class Ros2SdkInstallTest(unittest.TestCase):
    def test_sdk_metadata_selects_supported_environment_modes(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
cat > "${tmp}/ros2-release" <<'EOF'
Product Name = SiMa.ai ROS2 SDK
SDK Type = ros2-sdk
Platform Version = 2.1.2
EOF
printf '%s\n' 'SDK Version = 2.1.2_Palette_SDK_neat_main_deadbeef' > "${tmp}/elxr-release"
printf '%s\n' 'Product Name = Unknown SDK' > "${tmp}/unknown-release"
mkdir -p "${tmp}/sysroot"

ELXR_SDK_RELEASE_FILE="${tmp}/ros2-release"
printf 'ROS2=%s\n' "$(detect_env_mode)"
ELXR_SDK_RELEASE_FILE="${tmp}/elxr-release"
printf 'ELXR=%s\n' "$(detect_env_mode)"
ELXR_SDK_RELEASE_FILE="${tmp}/unknown-release"
unset SYSROOT || true
printf 'UNKNOWN=%s\n' "$(detect_env_mode)"
SYSROOT="${tmp}/sysroot"
printf 'OVERRIDE=%s\n' "$(detect_env_mode)"
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "ROS2=ros2-sdk",
                "ELXR=elxr-sdk",
                "UNKNOWN=unsupported",
                "OVERRIDE=elxr-sdk",
            ],
        )

    def test_ros2_sdk_platform_base_matches_manifest(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
cat > "${tmp}/sdk-release" <<'EOF'
SDK Type = ros2-sdk
Platform Version = 2.1.3~pre4678
Platform Base = 2.1.3
ROS2 SDK Version = main:deadbeef:20260806T225502Z
EOF
printf '%s\n' '{"platform-version":"2.1.3"}' > "${tmp}/manifest.json"
ELXR_SDK_RELEASE_FILE="${tmp}/sdk-release"
NEAT_PACKAGE_MANIFEST="${tmp}/manifest.json"
ENV_MODE=ros2-sdk
ensure_platform_compatible
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Platform compatibility verified: 2.1.3", result.stdout)

    def test_ros2_sdk_falls_back_to_platform_version_and_rejects_mismatch(
        self,
    ) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
cat > "${tmp}/sdk-release" <<'EOF'
SDK Type = ros2-sdk
Platform Version = 2.2.0
Version = 2.1.2
EOF
printf '%s\n' '{"platform-version":"2.1.2"}' > "${tmp}/manifest.json"
ELXR_SDK_RELEASE_FILE="${tmp}/sdk-release"
NEAT_PACKAGE_MANIFEST="${tmp}/manifest.json"
ENV_MODE=ros2-sdk
ensure_platform_compatible
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Package platform-version: 2.1.2", result.stderr)
        self.assertIn("Detected Platform Version: 2.2.0", result.stderr)

    def test_ros2_sdk_routes_only_to_native_install_without_pyneat(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
INSTALLER_TMP_DIRS=("${tmp}")
ENV_MODE=ros2-sdk
install_debs_in_ros2_sdk() { printf 'NATIVE_INSTALL\n'; }
install_agent_skills_for_current_user() { printf 'SKILLS=%s\n' "$1"; }
install_python_environment() { printf 'PYNEAT_PATH_USED\n'; return 99; }
install_debs_on_board() { printf 'BOARD_PATH_USED\n'; return 99; }
install_debs_into_sysroot() { printf 'SYSROOT_PATH_USED\n'; return 99; }
install_for_environment
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["NATIVE_INSTALL", "SKILLS=/usr/share/sima-neat/skills/sima-neat"],
        )

    def test_ros2_sdk_uses_zero_removal_native_apt_transaction(self) -> None:
        result = run_bash(
            r"""
source "$1"
DEBS=(./sima-neat.deb ./sima-neat-dev.deb ./neat-runtime.deb)
validate_ros2_sdk_native_host() { :; }
validate_single_sima_neat_package_pair() { :; }
validate_ros2_sdk_deb_architectures() { :; }
validate_ros2_sdk_tvm_runtime() { :; }
refresh_apt_metadata_for_board_install() { :; }
apt_package_database_is_healthy() { return 0; }
run_sudo() {
  if [[ " $* " == *" --simulate "* ]]; then
    printf 'SIMULATION_OK\n'
    return 0
  fi
  printf 'RUN:'
  printf ' <%s>' "$@"
  printf '\n'
}
install_ros2_sdk_tvm_runtime() { printf 'INSTALLED_TVM_RUNTIME\n'; }
repair_global_sima_neat_lib_links() { printf 'REPAIRED_LINKS\n'; }
verify_global_sima_neat_lib_links() { printf 'VERIFIED_LINKS\n'; }
verify_ros2_sdk_deb_packages_installed() { printf 'VERIFIED_PACKAGES\n'; }
install_debs_in_ros2_sdk
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIMULATION_OK", result.stdout)
        self.assertIn("INSTALLED_TVM_RUNTIME", result.stdout)
        self.assertIn("VERIFIED_PACKAGES", result.stdout)
        transaction = next(
            line for line in result.stdout.splitlines() if line.startswith("RUN:")
        )
        self.assertIn("<apt-get>", transaction)
        self.assertIn("<--no-remove>", transaction)
        self.assertIn("<--allow-downgrades>", transaction)
        self.assertNotIn("--fix-broken", transaction)

    def test_ros2_sdk_rejects_simulated_package_removal(self) -> None:
        result = run_bash(
            r"""
source "$1"
DEBS=(./sima-neat.deb ./sima-neat-dev.deb)
validate_ros2_sdk_native_host() { :; }
validate_single_sima_neat_package_pair() { :; }
validate_ros2_sdk_deb_architectures() { :; }
validate_ros2_sdk_tvm_runtime() { :; }
refresh_apt_metadata_for_board_install() { :; }
apt_package_database_is_healthy() { return 0; }
run_sudo() {
  if [[ " $* " == *" --simulate "* ]]; then
    printf '%s\n' 'Remv simaai-palette-modalix [2.1.2]'
    return 0
  fi
  printf 'MUTATED\n'
}
install_debs_in_ros2_sdk
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("planned package removal", result.stderr)
        self.assertNotIn("MUTATED", result.stdout)

    def test_ros2_sdk_rejects_missing_tvm_before_apt_transaction(self) -> None:
        result = run_bash(
            r"""
source "$1"
DEBS=(./sima-neat.deb ./sima-neat-dev.deb)
validate_ros2_sdk_native_host() { :; }
validate_single_sima_neat_package_pair() { :; }
validate_ros2_sdk_deb_architectures() { :; }
validate_ros2_sdk_tvm_runtime() {
  printf 'MISSING_TVM\n' >&2
  return 1
}
run_sudo() { printf 'APT_MUTATED\n'; }
install_debs_in_ros2_sdk
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("MISSING_TVM", result.stderr)
        self.assertNotIn("APT_MUTATED", result.stdout)

    def test_ros2_sdk_rejects_non_arm64_deb(self) -> None:
        result = run_bash(
            r"""
source "$1"
DEBS=(./sima-neat.deb ./sima-neat-dev.deb)
dpkg-deb() {
  case "$2" in
    ./sima-neat.deb) printf '%s\n' arm64 ;;
    ./sima-neat-dev.deb) printf '%s\n' amd64 ;;
    *) return 1 ;;
  esac
}
validate_ros2_sdk_deb_architectures
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be arm64 or all", result.stderr)
        self.assertIn("./sima-neat-dev.deb", result.stderr)

    def test_ros2_sdk_validates_native_arm64_debian_bookworm_host(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
cat > "${tmp}/os-release" <<'EOF'
ID=debian
VERSION_ID="12"
VERSION_CODENAME=bookworm
EOF
NEAT_OS_RELEASE_FILE="${tmp}/os-release"
apt-get() { :; }
dpkg-deb() { :; }
dpkg-query() { :; }
dpkg() { [[ "$1" == --print-architecture ]] && printf '%s\n' arm64; }
uname() { [[ "$1" == -m ]] && printf '%s\n' aarch64; }
validate_ros2_sdk_native_host
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ros2_sdk_rejects_non_arm64_host(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
cat > "${tmp}/os-release" <<'EOF'
ID=debian
VERSION_ID="12"
VERSION_CODENAME=bookworm
EOF
NEAT_OS_RELEASE_FILE="${tmp}/os-release"
apt-get() { :; }
dpkg-deb() { :; }
dpkg-query() { :; }
dpkg() { [[ "$1" == --print-architecture ]] && printf '%s\n' amd64; }
uname() { [[ "$1" == -m ]] && printf '%s\n' x86_64; }
validate_ros2_sdk_native_host
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires an ARM64 host environment", result.stderr)

    def test_ros2_sdk_installs_missing_tvm_runtime_from_matching_sysroot(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
mkdir -p "${tmp}/sysroot/usr/lib"
touch "${tmp}/sysroot/usr/lib/libtvm_runtime.so"
SYSROOT="${tmp}/sysroot"
run_sudo() {
  printf 'RUN:'
  printf ' <%s>' "$@"
  printf '\n'
}
readelf() { printf '%s\n' '  Machine: AArch64'; }
install_ros2_sdk_tvm_runtime
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "<install> <-d> <-m> <0755> </usr/lib/aarch64-linux-gnu>",
            result.stdout,
        )
        self.assertIn("libtvm_runtime.so>", result.stdout)


class NativeModalixRestoreTest(unittest.TestCase):
    def test_sdk_platform_version_prefers_pre_release_platform_base(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
cat > "${tmp}/sdk-release" <<'EOF'
SDK Profile = platform-cross
Platform Version = 2.1.3~pre4678
Platform Base = 2.1.3
Platform Channel = pre-release
SDK Version = 2.1.3~pre4678_Palette_SDK_neat_develop_4b9f4a1
EOF
read_sdk_platform_version "${tmp}/sdk-release"
sdk_platform_version_label "${tmp}/sdk-release"
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["2.1.3", "Platform Base"])

    def test_sdk_platform_version_preserves_legacy_sdk_release(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
cat > "${tmp}/sdk-release" <<'EOF'
SDK Version = 2.1.2.3_Palette_SDK_neat_release-2.1_3b4be39
eLXr Version = 2.1.2_release_neat_release-2.1_3b4be39
EOF
read_sdk_platform_version "${tmp}/sdk-release"
sdk_platform_version_label "${tmp}/sdk-release"
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["2.1.2.3", "SDK Version"])

    def test_pre_release_sdk_accepts_matching_package_platform_base(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
printf '%s\n' '{"platform-version":"2.1.3"}' > "${tmp}/manifest.json"
cat > "${tmp}/sdk-release" <<'EOF'
Platform Version = 2.1.3~pre4678
Platform Base = 2.1.3
SDK Version = 2.1.3~pre4678_Palette_SDK_neat_develop_4b9f4a1
EOF
ENV_MODE=elxr-sdk
NEAT_PACKAGE_MANIFEST="${tmp}/manifest.json"
ELXR_SDK_RELEASE_FILE="${tmp}/sdk-release"
NEAT_INSTALLER_SKIP_PLATFORM_CHECK=OFF
ensure_platform_compatible
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Platform compatibility verified: 2.1.3", result.stdout)

    def test_pre_release_sdk_rejects_different_package_platform_base(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
printf '%s\n' '{"platform-version":"2.1.2"}' > "${tmp}/manifest.json"
cat > "${tmp}/sdk-release" <<'EOF'
Platform Version = 2.1.3~pre4678
Platform Base = 2.1.3
SDK Version = 2.1.3~pre4678_Palette_SDK_neat_develop_4b9f4a1
EOF
ENV_MODE=elxr-sdk
NEAT_PACKAGE_MANIFEST="${tmp}/manifest.json"
ELXR_SDK_RELEASE_FILE="${tmp}/sdk-release"
NEAT_INSTALLER_SKIP_PLATFORM_CHECK=OFF
ensure_platform_compatible
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Detected Platform Base: 2.1.3", result.stderr)

    def test_board_transaction_accepts_exact_identity_preserving_replacement(
        self,
    ) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
replacement="${tmp}/neat-common.deb"
simulation="${tmp}/simulation.log"
touch "${replacement}"
printf '%s\n' 'Remv simaai-common [2.1.3~pre4678]' > "${simulation}"
dpkg-query() {
  printf '%s\n' '2.1.3~pre4678'
}
dpkg-deb() {
  [[ "$1" == -f ]] || return 2
  case "$3" in
    Provides) printf '%s\n' 'simaai-common (= 2.1.3~pre4678)' ;;
    Replaces) printf '%s\n' 'simaai-common' ;;
    Conflicts) printf '%s\n' 'simaai-common' ;;
    *) return 2 ;;
  esac
}
verify_simulated_package_removals "${simulation}" "${replacement}"
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Verified platform package replacements", result.stdout)
        self.assertIn("simaai-common=2.1.3~pre4678", result.stdout)

    def test_board_transaction_rejects_non_exact_replacement(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
replacement="${tmp}/neat-common.deb"
simulation="${tmp}/simulation.log"
touch "${replacement}"
printf '%s\n' 'Remv simaai-common [2.1.3~pre4678]' > "${simulation}"
dpkg-query() {
  printf '%s\n' '2.1.3~pre4678'
}
dpkg-deb() {
  [[ "$1" == -f ]] || return 2
  case "$3" in
    Provides) printf '%s\n' 'simaai-common (= 2.1.3)' ;;
    Replaces) printf '%s\n' 'simaai-common' ;;
    Conflicts) printf '%s\n' 'simaai-common' ;;
    *) return 2 ;;
  esac
}
verify_simulated_package_removals "${simulation}" "${replacement}"
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "without a bundled package that Provides its exact installed version",
            result.stderr,
        )

    def test_board_transaction_accepts_retired_neat_libcamera_removals(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
simulation="${tmp}/simulation.log"
printf '%s\n' \
  'Remv neat-libcamera [2.1.1~pre3348]' \
  'Remv neat-libcamera-dev [2.1.1~pre3348]' \
  'Remv neat-libcamera-tools [2.1.1~pre3348]' > "${simulation}"
verify_simulated_package_removals "${simulation}"
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_board_transaction_still_rejects_non_retired_removals(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
simulation="${tmp}/simulation.log"
printf '%s\n' 'Remv simaai-palette-modalix [2.1.3~pre4744]' > "${simulation}"
dpkg-query() {
  printf '%s\n' '2.1.3~pre4744'
}
verify_simulated_package_removals "${simulation}"
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "without a bundled package that Provides its exact installed version",
            result.stderr,
        )

    def test_heal_specs_stay_empty_on_boards_without_retired_packages(self) -> None:
        result = run_bash(
            r"""
source "$1"
dpkg-query() {
  if [[ "$2" == *Status-Abbrev* ]]; then
    printf 'un '
    return 1
  fi
  printf '%s\n' '2.1.3~pre4744'
}
collect_board_heal_specs
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_heal_specs_pin_palette_and_installed_identities_on_dirty_board(
        self,
    ) -> None:
        result = run_bash(
            r"""
source "$1"
dpkg-query() {
  case "$2" in
    *Status-Abbrev*)
      case "$3" in
        neat-libcamera | neat-libcamera-tools | simaai-palette-modalix)
          printf 'ii '
          return 0
          ;;
        *)
          printf 'un '
          return 1
          ;;
      esac
      ;;
    *)
      printf '%s\n' '2.1.3~pre4744'
      ;;
  esac
}
collect_board_heal_specs
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "simaai-palette-modalix=2.1.3~pre4744",
                "libcamera=2.1.3~pre4744",
                "libcamera-tools=2.1.3~pre4744",
            ],
        )

    def test_heal_specs_warn_and_stay_empty_when_palette_is_absent(self) -> None:
        result = run_bash(
            r"""
source "$1"
dpkg-query() {
  case "$2" in
    *Status-Abbrev*)
      if [[ "$3" == simaai-palette-modalix ]]; then
        printf 'un '
        return 1
      fi
      printf 'ii '
      return 0
      ;;
    *)
      printf '%s\n' '2.1.3~pre4744'
      ;;
  esac
}
collect_board_heal_specs
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("simaai-palette-modalix version is unavailable", result.stderr)


class DispatcherMigrationTest(unittest.TestCase):
    def test_migration_moves_unowned_global_and_backup_outside_loader_dir(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
global="${tmp}/loader"
quarantine="${tmp}/quarantine"
mkdir -p "${global}"
printf stale > "${global}/libneatdispatchercore.so.bak-20260705"
ln -s libneatdispatchercore.so.bak-20260705 "${global}/libneatdispatchercore.so"
dispatcher_global_lib_dir() { printf '%s\n' "${global}"; }
dispatcher_quarantine_root() { printf '%s\n' "${quarantine}"; }
dpkg-query() { return 1; }
run_sudo() {
  [[ "$1" == ldconfig ]] && return 0
  "$@"
}
migrate_stale_global_dispatcher_libs
[[ ! -e "${global}/libneatdispatchercore.so" && ! -L "${global}/libneatdispatchercore.so" ]]
[[ ! -e "${global}/libneatdispatchercore.so.bak-20260705" ]]
[[ "$(find "${quarantine}" -type f -name 'libneatdispatchercore.so.bak-20260705' | wc -l)" -eq 1 ]]
[[ "$(find "${quarantine}" -type l -name 'libneatdispatchercore.so' | wc -l)" -eq 1 ]]
printf 'MIGRATED\n'
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MIGRATED", result.stdout)
        self.assertNotIn("Linking", result.stdout)

    def test_migration_refuses_package_owned_global_dispatcher(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
mkdir -p "${tmp}/loader"
touch "${tmp}/loader/libneatdispatchercore.so"
dispatcher_global_lib_dir() { printf '%s\n' "${tmp}/loader"; }
dispatcher_quarantine_root() { printf '%s\n' "${tmp}/quarantine"; }
dpkg-query() { printf 'legacy-runtime: %s\n' "$2"; }
run_sudo() { "$@"; }
migrate_stale_global_dispatcher_libs
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("package-owned global dispatcher", result.stderr)

    def test_verifies_versioned_private_dispatcher_and_package_ownership(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
global="${tmp}/loader"
runtime="${global}/neat/runtime"
mkdir -p "${runtime}"
touch "${runtime}/libneatdispatchercore.so.0.4.0"
dispatcher_global_lib_dir() { printf '%s\n' "${global}"; }
dispatcher_private_runtime_dir() { printf '%s\n' "${runtime}"; }
readelf() { printf '%s\n' ' 0x000000000000000e (SONAME) Library soname: [libneatdispatchercore.so.0.4.0]'; }
dpkg-query() { printf 'neat-runtime: %s\n' "$2"; }
verify_private_dispatcher_runtime
printf 'PRIVATE_OK\n'
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PRIVATE_OK", result.stdout)
        self.assertIn("versioned package-owned dispatcher", result.stdout)


class DevKitRecoveryDispatcherTest(unittest.TestCase):
    def test_recovery_quarantines_every_global_dispatcher_without_alias(self) -> None:
        result = run_bash(
            r"""
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
mkdir -p "${tmp}/loader"
printf stale > "${tmp}/loader/libneatdispatchercore.so.bak-20260705"
ln -s libneatdispatchercore.so.bak-20260705 \
  "${tmp}/loader/libneatdispatchercore.so"
export NEAT_RECOVERY_DISPATCHER_GLOBAL_LIB_DIR="${tmp}/loader"
export NEAT_RECOVERY_DISPATCHER_QUARANTINE_DIR="${tmp}/quarantine"
export NEAT_RECOVERY_FUNCTIONS_ONLY=ON
source "$1"
run_step() {
  shift
  "$@"
}
dpkg-query() { return 1; }
ldconfig() { :; }
quarantine_stale_global_dispatcher_libs
[[ ! -e "${tmp}/loader/libneatdispatchercore.so" ]]
[[ ! -L "${tmp}/loader/libneatdispatchercore.so" ]]
[[ ! -e "${tmp}/loader/libneatdispatchercore.so.bak-20260705" ]]
[[ -L "${tmp}/quarantine/libneatdispatchercore.so" ]]
[[ -f "${tmp}/quarantine/libneatdispatchercore.so.bak-20260705" ]]
[[ "$(find "${tmp}/loader" -name 'libneatdispatchercore.so*' | wc -l)" -eq 0 ]]
printf 'RECOVERY_MIGRATED\n'
""",
            RECOVERY,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RECOVERY_MIGRATED", result.stdout)
        self.assertIn("no global alias was created", result.stdout)

    def test_recovery_refuses_package_owned_global_dispatcher(self) -> None:
        result = run_bash(
            r"""
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
mkdir -p "${tmp}/loader"
touch "${tmp}/loader/libneatdispatchercore.so"
export NEAT_RECOVERY_DISPATCHER_GLOBAL_LIB_DIR="${tmp}/loader"
export NEAT_RECOVERY_DISPATCHER_QUARANTINE_DIR="${tmp}/quarantine"
export NEAT_RECOVERY_FUNCTIONS_ONLY=ON
source "$1"
run_step() {
  shift
  "$@"
}
dpkg-query() { printf 'legacy-runtime: %s\n' "$2"; }
ldconfig() { :; }
! quarantine_stale_global_dispatcher_libs
[[ -f "${tmp}/loader/libneatdispatchercore.so" ]]
[[ ! -e "${tmp}/quarantine" ]]
printf 'RECOVERY_REFUSED\n'
""",
            RECOVERY,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RECOVERY_REFUSED", result.stdout)
        self.assertIn("package-owned global dispatcher", result.stderr)


RECOVERY_ORDER_HARNESS = r"""
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
calls="${tmp}/calls"
: > "${calls}"
export NEAT_RECOVERY_FUNCTIONS_ONLY=ON
source "$1"

# Record the step labels instead of running them; their order is the contract.
run_step() { printf '%s\n' "$1" >> "${calls}"; }
run_optional_service_step() { printf '%s\n' "$1" >> "${calls}"; }
empty_coprocessing() { :; }
cleanup_tmp_sima_if_root_low_space() { :; }
systemctl() { return 0; }
sleep() { :; }

line_of() { grep -n -- "$1" "${calls}" | head -1 | cut -d: -f1; }
"""


class DevKitRecoveryOrderingTest(unittest.TestCase):
    """The M4 must only be booted while mlashmcomplex is alive (#659)."""

    def test_m4_boots_before_appcomplex_is_stopped(self) -> None:
        result = run_bash(
            RECOVERY_ORDER_HARNESS
            + r"""
pgrep() { return 0; }   # appcomplex already running

recover_devkit_runtime

boot="$(line_of 'remoteproc1 start')"
stop="$(line_of 'stop simaai-appcomplex.service')"
init="$(line_of 'init_mla_memory')"
restart="$(line_of 'restart simaai-appcomplex.service')"

(( boot < stop ))    || { printf 'M4 booted after appcomplex stop\n' >&2; exit 1; }
(( stop < init ))    || { printf 'init_mla_memory ran while appcomplex held the mailbox\n' >&2; exit 1; }
(( init < restart )) || { printf 'appcomplex restarted before init_mla_memory\n' >&2; exit 1; }
printf 'RECOVERY_ORDER_OK\n'
""",
            RECOVERY,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RECOVERY_ORDER_OK", result.stdout)

    def test_recovery_restarts_a_down_appcomplex_before_booting_the_m4(self) -> None:
        result = run_bash(
            RECOVERY_ORDER_HARNESS
            + r"""
# Down on the first probe, up once the start step has run.
probe_count=0
pgrep() {
  probe_count=$(( probe_count + 1 ))
  (( probe_count > 1 ))
}

recover_devkit_runtime

start="$(line_of 'start simaai-appcomplex.service')"
reset="$(line_of 'clear simaai-appcomplex.service start-limit state')"
boot="$(line_of 'remoteproc1 start')"

[[ -n "${reset}" ]] || { printf 'start-limit state was never cleared\n' >&2; exit 1; }
(( reset < start )) || { printf 'start attempted before clearing the start limit\n' >&2; exit 1; }
(( start < boot ))  || { printf 'M4 booted before appcomplex was restored\n' >&2; exit 1; }
printf 'RECOVERY_RESTORED_APPCOMPLEX\n'
""",
            RECOVERY,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RECOVERY_RESTORED_APPCOMPLEX", result.stdout)

    def test_recovery_refuses_to_boot_the_m4_when_appcomplex_stays_down(self) -> None:
        result = run_bash(
            RECOVERY_ORDER_HARNESS
            + r"""
pgrep() { return 1; }   # never comes up

if recover_devkit_runtime; then
  printf 'recovery reported success with appcomplex down\n' >&2
  exit 1
fi

if grep -q 'remoteproc' "${calls}"; then
  printf 'remoteproc was touched with appcomplex down\n' >&2
  exit 1
fi
printf 'RECOVERY_REFUSED_M4_BOOT\n'
""",
            RECOVERY,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RECOVERY_REFUSED_M4_BOOT", result.stdout)
        self.assertIn("refusing to boot the M4", result.stderr)

    def test_recovery_skips_the_precondition_when_appcomplex_is_not_installed(
        self,
    ) -> None:
        result = run_bash(
            RECOVERY_ORDER_HARNESS
            + r"""
systemctl() { return 1; }   # unit not installed on this image
pgrep() { return 1; }

recover_devkit_runtime

boot="$(line_of 'remoteproc1 start')"
[[ -n "${boot}" ]] || { printf 'recovery stalled on an image without appcomplex\n' >&2; exit 1; }
printf 'RECOVERY_SKIPPED_PRECONDITION\n'
""",
            RECOVERY,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RECOVERY_SKIPPED_PRECONDITION", result.stdout)
        self.assertIn("is not installed on this devkit image", result.stdout)


class SimaNeatLinkRepairTest(unittest.TestCase):
    def test_sdk_sysroot_rejects_multiple_core_package_pairs(self) -> None:
        result = run_bash(
            r"""
source "$1"
DEBS=(core-old.deb core-new.deb dev-new.deb)
dpkg-deb() {
  [[ "$1" == -f ]] || return 2
  case "$2:$3" in
    core-old.deb:Package|core-new.deb:Package) printf '%s\n' sima-neat ;;
    dev-new.deb:Package) printf '%s\n' sima-neat-dev ;;
    core-old.deb:Version) printf '%s\n' 0.2.0 ;;
    core-new.deb:Version|dev-new.deb:Version) printf '%s\n' 0.3.0 ;;
    *) return 2 ;;
  esac
}
validate_single_sima_neat_package_pair
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "requires exactly one sima-neat and one sima-neat-dev package",
            result.stderr,
        )
        self.assertIn("sima-neat packages:     2", result.stderr)

    def test_sdk_sysroot_rejects_mismatched_core_package_versions(self) -> None:
        result = run_bash(
            r"""
source "$1"
DEBS=(core.deb dev.deb)
dpkg-deb() {
  [[ "$1" == -f ]] || return 2
  case "$2:$3" in
    core.deb:Package) printf '%s\n' sima-neat ;;
    dev.deb:Package) printf '%s\n' sima-neat-dev ;;
    core.deb:Version) printf '%s\n' 0.3.0+core ;;
    dev.deb:Version) printf '%s\n' 0.3.0+dev ;;
    *) return 2 ;;
  esac
}
validate_single_sima_neat_package_pair
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("package versions do not match", result.stderr)
        self.assertIn("0.3.0+core", result.stderr)
        self.assertIn("0.3.0+dev", result.stderr)

    def test_sdk_sysroot_accepts_one_matching_core_package_pair(self) -> None:
        result = run_bash(
            r"""
source "$1"
DEBS=(core.deb dev.deb unrelated.deb)
dpkg-deb() {
  [[ "$1" == -f ]] || return 2
  case "$2:$3" in
    core.deb:Package) printf '%s\n' sima-neat ;;
    dev.deb:Package) printf '%s\n' sima-neat-dev ;;
    unrelated.deb:Package) printf '%s\n' neat-runtime ;;
    core.deb:Version|dev.deb:Version) printf '%s\n' 0.3.0 ;;
    unrelated.deb:Version) printf '%s\n' 1.0 ;;
    *) return 2 ;;
  esac
}
validate_single_sima_neat_package_pair
printf 'PAIR_OK\n'
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PAIR_OK", result.stdout)

    def test_sdk_sysroot_preserves_current_bundle_compatibility_link(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
mkdir -p "${tmp}/usr/lib"
lib_dir="${tmp}/usr/lib"
touch "${lib_dir}/libsima_neat.so.2.1.2"
ln -s libsima_neat.so.2.1.2 "${lib_dir}/libsima_neat.so.3"
ln -s libsima_neat.so.3 "${lib_dir}/libsima_neat.so"
ln -s libsima_neat.so.2.1.2 "${lib_dir}/libsima_neat.so.2"

collect_current_bundle_sima_neat_lib_paths() {
  local sysroot="$1"
  local -n out="$2"
  out=(
    "${sysroot}/usr/lib/libsima_neat.so.2.1.2"
    "${sysroot}/usr/lib/libsima_neat.so.3"
    "${sysroot}/usr/lib/libsima_neat.so.2"
    "${sysroot}/usr/lib/libsima_neat.so"
  )
}
read_sima_neat_elf_soname() { printf '%s\n' 'libsima_neat.so.3'; }
run_sudo() { "$@"; }

repair_sysroot_sima_neat_libs "${tmp}"
[[ "$(readlink "${lib_dir}/libsima_neat.so.2")" == 'libsima_neat.so.2.1.2' ]]
! compgen -G "${lib_dir}/libsima_neat.so.2.bak-neat-installer-*" >/dev/null
printf 'SDK_COMPAT_OK\n'
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SDK_COMPAT_OK", result.stdout)
        self.assertNotIn(
            "Quarantining stale SDK sysroot libsima_neat path", result.stdout
        )

    def test_sdk_sysroot_quarantines_libraries_not_owned_by_current_bundle(
        self,
    ) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
mkdir -p "${tmp}/usr/lib"
lib_dir="${tmp}/usr/lib"
touch "${lib_dir}/libsima_neat.so.2.1.2"
ln -s libsima_neat.so.2.1.2 "${lib_dir}/libsima_neat.so.3"
ln -s libsima_neat.so.3 "${lib_dir}/libsima_neat.so"
touch "${lib_dir}/libsima_neat.so.2.0.0"
ln -s libsima_neat.so.2.0.0 "${lib_dir}/libsima_neat.so.2"

collect_current_bundle_sima_neat_lib_paths() {
  local sysroot="$1"
  local -n out="$2"
  out=(
    "${sysroot}/usr/lib/libsima_neat.so.2.1.2"
    "${sysroot}/usr/lib/libsima_neat.so.3"
    "${sysroot}/usr/lib/libsima_neat.so"
  )
}
read_sima_neat_elf_soname() { printf '%s\n' 'libsima_neat.so.3'; }
run_sudo() { "$@"; }

repair_sysroot_sima_neat_libs "${tmp}"
[[ "$(readlink -f "${lib_dir}/libsima_neat.so")" == "${lib_dir}/libsima_neat.so.2.1.2" ]]
[[ ! -e "${lib_dir}/libsima_neat.so.2" && ! -L "${lib_dir}/libsima_neat.so.2" ]]
[[ ! -e "${lib_dir}/libsima_neat.so.2.0.0" ]]
compgen -G "${lib_dir}/libsima_neat.so.2.bak-neat-installer-*" >/dev/null
compgen -G "${lib_dir}/libsima_neat.so.2.0.0.bak-neat-installer-*" >/dev/null
printf 'SDK_ABI3_OK\n'
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SDK_ABI3_OK", result.stdout)
        self.assertEqual(
            result.stdout.count("Quarantining stale SDK sysroot libsima_neat path"),
            2,
        )

    def test_abi3_package_manifest_drives_links_and_quarantines_unowned_abi2(
        self,
    ) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
touch "${tmp}/libsima_neat.so.2.1.2"
ln -s libsima_neat.so.2.1.2 "${tmp}/libsima_neat.so.3"
ln -s libsima_neat.so.3 "${tmp}/libsima_neat.so"
ln -s libsima_neat.so.2.1.2 "${tmp}/libsima_neat.so.2"
soname_inode="$(stat -c '%i' "${tmp}/libsima_neat.so.3")"

sima_neat_global_lib_dir() { printf '%s\n' "${tmp}"; }
dpkg-query() {
  case "$1" in
    -L)
      printf '%s\n' \
        "${tmp}/libsima_neat.so.2.1.2" \
        "${tmp}/libsima_neat.so.3"
      ;;
    -S)
      case "$2" in
        "${tmp}/libsima_neat.so.2.1.2"|"${tmp}/libsima_neat.so.3"|"${tmp}/libsima_neat.so")
          printf 'sima-neat: %s\n' "$2"
          ;;
        *) return 1 ;;
      esac
      ;;
    *) return 1 ;;
  esac
}
read_sima_neat_elf_soname() { printf '%s\n' 'libsima_neat.so.3'; }
run_sudo() { "$@"; }

repair_global_sima_neat_lib_links
verify_global_sima_neat_lib_links
[[ "$(stat -c '%i' "${tmp}/libsima_neat.so.3")" == "${soname_inode}" ]]
[[ "$(readlink "${tmp}/libsima_neat.so")" == 'libsima_neat.so.3' ]]
[[ ! -e "${tmp}/libsima_neat.so.2" && ! -L "${tmp}/libsima_neat.so.2" ]]
compgen -G "${tmp}/libsima_neat.so.2.bak-neat-installer-*" >/dev/null
printf 'ABI3_OK\n'
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ABI3_OK", result.stdout)
        self.assertIn("Quarantining stale unowned libsima_neat path", result.stdout)
        self.assertNotIn("Repairing " + "/usr/lib/libsima_neat.so.2", result.stdout)

    def test_package_owned_compatibility_soname_is_never_quarantined(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
touch "${tmp}/libsima_neat.so.2.1.2"
ln -s libsima_neat.so.2.1.2 "${tmp}/libsima_neat.so.3"
ln -s libsima_neat.so.3 "${tmp}/libsima_neat.so"
ln -s libsima_neat.so.2.1.2 "${tmp}/libsima_neat.so.2"

sima_neat_global_lib_dir() { printf '%s\n' "${tmp}"; }
dpkg-query() {
  case "$1" in
    -L)
      printf '%s\n' \
        "${tmp}/libsima_neat.so.2.1.2" \
        "${tmp}/libsima_neat.so.3"
      ;;
    -S)
      printf 'test-package: %s\n' "$2"
      ;;
    *) return 1 ;;
  esac
}
read_sima_neat_elf_soname() { printf '%s\n' 'libsima_neat.so.3'; }
run_sudo() { "$@"; }

repair_global_sima_neat_lib_links
[[ "$(readlink "${tmp}/libsima_neat.so.2")" == 'libsima_neat.so.2.1.2' ]]
! compgen -G "${tmp}/libsima_neat.so.2.bak-neat-installer-*" >/dev/null
printf 'OWNED_OK\n'
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OWNED_OK", result.stdout)
        self.assertIn(
            "Preserving package-owned libsima_neat compatibility link", result.stdout
        )

    def test_wrong_package_owned_manifest_links_are_repaired_without_quarantine(
        self,
    ) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
touch "${tmp}/libsima_neat.so.2.1.2"
ln -s missing-libsima-neat "${tmp}/libsima_neat.so.3"
ln -s libsima_neat.so.2 "${tmp}/libsima_neat.so"
ln -s libsima_neat.so.2.1.2 "${tmp}/libsima_neat.so.2"

sima_neat_global_lib_dir() { printf '%s\n' "${tmp}"; }
dpkg-query() {
  case "$1" in
    -L)
      printf '%s\n' \
        "${tmp}/libsima_neat.so.2.1.2" \
        "${tmp}/libsima_neat.so.3"
      ;;
    -S) printf 'test-package: %s\n' "$2" ;;
    *) return 1 ;;
  esac
}
read_sima_neat_elf_soname() { printf '%s\n' 'libsima_neat.so.3'; }
run_sudo() { "$@"; }

repair_global_sima_neat_lib_links
verify_global_sima_neat_lib_links
[[ "$(readlink "${tmp}/libsima_neat.so.3")" == 'libsima_neat.so.2.1.2' ]]
[[ "$(readlink "${tmp}/libsima_neat.so")" == 'libsima_neat.so.3' ]]
[[ "$(readlink "${tmp}/libsima_neat.so.2")" == 'libsima_neat.so.2.1.2' ]]
! compgen -G "${tmp}/libsima_neat.so*.bak-neat-installer-*" >/dev/null
printf 'OWNED_REPAIR_OK\n'
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OWNED_REPAIR_OK", result.stdout)
        self.assertEqual(result.stdout.count("Repairing package-owned symlink"), 2)

    def test_elf_soname_must_match_packaged_soname_link(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
touch "${tmp}/libsima_neat.so.2.1.2"
ln -s libsima_neat.so.2.1.2 "${tmp}/libsima_neat.so.3"
ln -s libsima_neat.so.3 "${tmp}/libsima_neat.so"

sima_neat_global_lib_dir() { printf '%s\n' "${tmp}"; }
dpkg-query() {
  case "$1" in
    -L)
      printf '%s\n' \
        "${tmp}/libsima_neat.so.2.1.2" \
        "${tmp}/libsima_neat.so.3"
      ;;
    -S) printf 'sima-neat: %s\n' "$2" ;;
    *) return 1 ;;
  esac
}
read_sima_neat_elf_soname() { printf '%s\n' 'libsima_neat.so.2'; }
run_sudo() { "$@"; }

repair_global_sima_neat_lib_links
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Packaged libsima_neat SONAME does not match its package manifest",
            result.stderr,
        )


def _take_path_option(flag: str) -> Path | None:
    """Pop `--flag <path>` out of argv; under CTest neither script sits at __file__."""
    if flag not in sys.argv:
        return None
    index = sys.argv.index(flag)
    try:
        value = Path(sys.argv[index + 1]).resolve()
    except IndexError as exc:
        raise SystemExit(f"{flag} requires a path") from exc
    del sys.argv[index : index + 2]
    return value


if __name__ == "__main__":
    INSTALLER = _take_path_option("--installer") or INSTALLER
    RECOVERY = _take_path_option("--recovery") or RECOVERY
    unittest.main()
