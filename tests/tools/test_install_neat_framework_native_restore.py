#!/usr/bin/env python3
"""Focused tests for native Modalix package recovery in the installer."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "tools" / "install_neat_framework.sh"


def run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script, "bash", str(INSTALLER)],
        check=False,
        text=True,
        capture_output=True,
    )


class NativeModalixRestoreTest(unittest.TestCase):
    def test_board_install_combines_native_repair_and_local_debs_atomically(self) -> None:
        result = run_bash(
            r'''
source "$1"
DEBS=(./neat-gst-plugins_fixed.deb ./sima-neat_fixed.deb ./libcamera_2.1.1_arm64.deb)
prepare_debs_for_board_install() { :; }
refresh_apt_metadata_for_board_install() { :; }
stop_board_runtime_before_install() { :; }
apt_package_database_is_healthy() { return 0; }
native_modalix_repair_is_required() { return 0; }
native_modalix_restore_specs() {
  local -n out="$1"
  out=(./libcamera_2.1.1_arm64.deb simaai-memory-lib=2.1.1 \
       simaai-gst-plugins simaai-palette-modalix=2.1.2)
}
run_sudo() {
  printf 'APT:'
  printf ' <%s>' "$@"
  printf '\n'
}
repair_stale_global_dispatcher_lib() { :; }
repair_global_sima_neat_lib_links() { :; }
verify_global_sima_neat_lib_links() { :; }
activate_board_runtime_after_install() { :; }
restart_board_codec_services() { :; }
verify_board_codec_services() { :; }
verify_board_runtime_services() { :; }

install_debs_on_board
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        apt_lines = [line for line in result.stdout.splitlines() if line.startswith("APT:")]
        self.assertEqual(len(apt_lines), 1, result.stdout)
        transaction = apt_lines[0]
        for required in (
            "<./neat-gst-plugins_fixed.deb>",
            "<./sima-neat_fixed.deb>",
            "<./libcamera_2.1.1_arm64.deb>",
            "<simaai-memory-lib=2.1.1>",
            "<simaai-gst-plugins>",
            "<simaai-palette-modalix=2.1.2>",
            "<--allow-downgrades>",
            "<--no-remove>",
        ):
            self.assertIn(required, transaction)
        self.assertEqual(transaction.count("<./libcamera_2.1.1_arm64.deb>"), 1)

    def test_restore_transaction_pins_palette_dependencies_and_installed_dev_packages(self) -> None:
        result = run_bash(
            r'''
source "$1"
DEBS=()
apt-cache() {
  case "$1:$2" in
    policy:simaai-palette-modalix)
      printf '%s\n' \
        'simaai-palette-modalix:' \
        '  Installed: (none)' \
        '  Candidate: 2.1.2'
      ;;
    show:simaai-palette-modalix=2.1.2)
      cat <<'EOF'
Package: simaai-palette-modalix
Version: 2.1.2
Depends: appcomplex (= 2.1.1), libcamera (= 2.1.1),
 libcamera-tools (= 2.1.1), simaai-memory-lib (= 2.1.1)
Description: test palette
EOF
      ;;
    show:simaai-memory-lib=2.1.1|show:simaai-memory-lib-dev=2.1.1)
      printf 'Package: %s\nVersion: 2.1.1\n' "${2%%=*}"
      ;;
    *) return 100 ;;
  esac
}
local_deb_for_exact_package() {
  case "$1:$2" in
    libcamera:2.1.1) printf '%s\n' './libcamera_2.1.1_arm64.deb' ;;
    libcamera-dev:2.1.1) printf '%s\n' './libcamera-dev_2.1.1_arm64.deb' ;;
    libcamera-tools:2.1.1) printf '%s\n' './libcamera-tools_2.1.1_arm64.deb' ;;
    *) return 1 ;;
  esac
}
deb_package_is_present() {
  [[ "$1" == libcamera-dev || "$1" == simaai-memory-lib-dev ]]
}

native_modalix_restore_specs specs
printf '%s\n' "${specs[@]}"
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "./libcamera_2.1.1_arm64.deb",
                "./libcamera-dev_2.1.1_arm64.deb",
                "./libcamera-tools_2.1.1_arm64.deb",
                "simaai-memory-lib=2.1.1",
                "simaai-memory-lib-dev=2.1.1",
                "simaai-gst-plugins",
                "simaai-palette-modalix=2.1.2",
            ],
        )

    def test_private_same_name_version_forces_repair(self) -> None:
        result = run_bash(
            r'''
source "$1"
deb_package_is_installed() { return 0; }
deb_package_installed_version() {
  if [[ "$1" == simaai-memory-lib ]]; then
    printf '%s\n' '2.1.1+neat1'
  else
    printf '%s\n' '2.1.1'
  fi
}
native_modalix_repair_is_required
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_canonical_native_state_does_not_force_repair(self) -> None:
        result = run_bash(
            r'''
source "$1"
deb_package_is_installed() { return 0; }
deb_package_installed_version() { printf '%s\n' '2.1.1'; }
if native_modalix_repair_is_required; then
  exit 99
fi
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_exact_dependency_is_rejected(self) -> None:
        result = run_bash(
            r'''
source "$1"
DEBS=()
apt-cache() {
  case "$1:$2" in
    policy:simaai-palette-modalix) printf '%s\n' '  Candidate: 2.1.2' ;;
    show:simaai-palette-modalix=2.1.2)
      printf '%s\n' \
        'Package: simaai-palette-modalix' \
        'Version: 2.1.2' \
        'Depends: libcamera (= 2.1.1), libcamera-tools (= 2.1.1), simaai-memory-lib (= 2.1.1)'
      ;;
    show:libcamera=2.1.1|show:libcamera-tools=2.1.1)
      printf 'Package: %s\nVersion: 2.1.1\n' "${2%%=*}"
      ;;
    *) return 100 ;;
  esac
}
deb_package_is_present() { return 1; }
native_modalix_restore_specs specs
'''
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Required canonical Modalix package is unavailable locally and from apt: simaai-memory-lib=2.1.1",
            result.stderr,
        )


class SimaNeatLinkRepairTest(unittest.TestCase):
    def test_sdk_sysroot_rejects_multiple_core_package_pairs(self) -> None:
        result = run_bash(
            r'''
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
'''
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "requires exactly one sima-neat and one sima-neat-dev package",
            result.stderr,
        )
        self.assertIn("sima-neat packages:     2", result.stderr)

    def test_sdk_sysroot_rejects_mismatched_core_package_versions(self) -> None:
        result = run_bash(
            r'''
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
'''
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("package versions do not match", result.stderr)
        self.assertIn("0.3.0+core", result.stderr)
        self.assertIn("0.3.0+dev", result.stderr)

    def test_sdk_sysroot_accepts_one_matching_core_package_pair(self) -> None:
        result = run_bash(
            r'''
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
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PAIR_OK", result.stdout)

    def test_sdk_sysroot_preserves_current_bundle_compatibility_link(self) -> None:
        result = run_bash(
            r'''
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
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SDK_COMPAT_OK", result.stdout)
        self.assertNotIn(
            "Quarantining stale SDK sysroot libsima_neat path", result.stdout
        )

    def test_sdk_sysroot_quarantines_libraries_not_owned_by_current_bundle(self) -> None:
        result = run_bash(
            r'''
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
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SDK_ABI3_OK", result.stdout)
        self.assertEqual(
            result.stdout.count("Quarantining stale SDK sysroot libsima_neat path"),
            2,
        )

    def test_abi3_package_manifest_drives_links_and_quarantines_unowned_abi2(self) -> None:
        result = run_bash(
            r'''
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
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ABI3_OK", result.stdout)
        self.assertIn("Quarantining stale unowned libsima_neat path", result.stdout)
        self.assertNotIn("Repairing " + "/usr/lib/libsima_neat.so.2", result.stdout)

    def test_package_owned_compatibility_soname_is_never_quarantined(self) -> None:
        result = run_bash(
            r'''
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
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OWNED_OK", result.stdout)
        self.assertIn("Preserving package-owned libsima_neat compatibility link", result.stdout)

    def test_wrong_package_owned_manifest_links_are_repaired_without_quarantine(self) -> None:
        result = run_bash(
            r'''
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
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OWNED_REPAIR_OK", result.stdout)
        self.assertEqual(result.stdout.count("Repairing package-owned symlink"), 2)

    def test_elf_soname_must_match_packaged_soname_link(self) -> None:
        result = run_bash(
            r'''
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
'''
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Packaged libsima_neat SONAME does not match its package manifest",
            result.stderr,
        )


if __name__ == "__main__":
    if "--installer" in sys.argv:
        index = sys.argv.index("--installer")
        try:
            INSTALLER = Path(sys.argv[index + 1]).resolve()
        except IndexError as exc:
            raise SystemExit("--installer requires a path") from exc
        del sys.argv[index : index + 2]
    unittest.main()
