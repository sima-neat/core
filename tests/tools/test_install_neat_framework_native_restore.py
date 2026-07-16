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
    def test_board_install_keeps_memory_out_of_broad_native_transaction(self) -> None:
        result = run_bash(
            r'''
source "$1"
DEBS=(./simaai-memory-lib_2.1.1_arm64.deb \
      ./simaai-memory-lib-dev_2.1.1_arm64.deb \
      ./neat-gst-plugins_fixed.deb ./sima-neat_fixed.deb ./libcamera_2.1.1_arm64.deb)
prepare_debs_for_board_install() { :; }
refresh_apt_metadata_for_board_install() { :; }
stop_board_runtime_before_install() { :; }
apt_package_database_is_healthy() { return 0; }
install_local_simaai_memory_transaction() {
  SIMAAI_MEMORY_TRANSACTION_COMPLETE=1
  DEBS=(./neat-gst-plugins_fixed.deb ./sima-neat_fixed.deb ./libcamera_2.1.1_arm64.deb)
}
native_modalix_repair_is_required() { return 0; }
native_modalix_restore_specs() {
  local -n out="$1"
  out=(./libcamera_2.1.1_arm64.deb simaai-gst-plugins simaai-palette-modalix=2.1.2)
}
run_sudo() {
  printf 'APT:'
  printf ' <%s>' "$@"
  printf '\n'
}
complete_board_install_after_packages() { :; }

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
            "<simaai-gst-plugins>",
            "<simaai-palette-modalix=2.1.2>",
            "<--allow-downgrades>",
        ):
            self.assertIn(required, transaction)
        self.assertNotIn("simaai-memory-lib", transaction)
        self.assertNotIn("<--no-remove>", transaction)
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


class SimaaiMemoryTransactionTest(unittest.TestCase):
    def test_collect_discovers_palette_exact_revision_and_local_pair(self) -> None:
        result = run_bash(
            r'''
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
runtime="${tmp}/simaai-memory-lib_2.1.1-4_arm64.deb"
dev="${tmp}/simaai-memory-lib-dev_2.1.1-4_arm64.deb"
touch "${runtime}" "${dev}"
DEBS=("${runtime}" "${dev}" other.deb)
palette_required_simaai_memory_version() { printf '%s\n' '2.1.1-4'; }
board_debian_architecture() { printf '%s\n' arm64; }
dpkg-deb() {
  [[ "$1" == -f ]] || return 2
  file="$(basename "$2")"
  case "${file}:$3" in
    simaai-memory-lib_2.1.1-4_arm64.deb:Package) printf '%s\n' simaai-memory-lib ;;
    simaai-memory-lib-dev_2.1.1-4_arm64.deb:Package) printf '%s\n' simaai-memory-lib-dev ;;
    *:Version) printf '%s\n' 2.1.1-4 ;;
    *:Architecture) printf '%s\n' arm64 ;;
    simaai-memory-lib-dev_2.1.1-4_arm64.deb:Depends)
      printf '%s\n' 'libc6, simaai-memory-lib (= 2.1.1-4)'
      ;;
    *) return 2 ;;
  esac
}
collect_local_simaai_memory_debs
printf 'VERSION=%s\n' "${SIMAAI_MEMORY_REQUIRED_VERSION}"
printf 'RUNTIME=%s\n' "${SIMAAI_MEMORY_RUNTIME_DEB}"
printf 'DEV=%s\n' "${SIMAAI_MEMORY_DEV_DEB}"
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VERSION=2.1.1-4", result.stdout)
        self.assertIn("simaai-memory-lib_2.1.1-4_arm64.deb", result.stdout)
        self.assertIn("simaai-memory-lib-dev_2.1.1-4_arm64.deb", result.stdout)

    def test_collect_rejects_local_revision_not_required_by_palette(self) -> None:
        result = run_bash(
            r'''
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
runtime="${tmp}/runtime.deb"
dev="${tmp}/dev.deb"
touch "${runtime}" "${dev}"
DEBS=("${runtime}" "${dev}")
palette_required_simaai_memory_version() { printf '%s\n' '2.1.1'; }
board_debian_architecture() { printf '%s\n' arm64; }
dpkg-deb() {
  [[ "$1" == -f ]] || return 2
  case "$(basename "$2"):$3" in
    runtime.deb:Package) printf '%s\n' simaai-memory-lib ;;
    dev.deb:Package) printf '%s\n' simaai-memory-lib-dev ;;
    *:Version) printf '%s\n' 2.1.1-1 ;;
    *:Architecture) printf '%s\n' arm64 ;;
    dev.deb:Depends) printf '%s\n' 'simaai-memory-lib (= 2.1.1-1)' ;;
    *) return 2 ;;
  esac
}
collect_local_simaai_memory_debs
'''
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("palette requires exact revision 2.1.1", result.stderr)

    def test_payload_validation_records_final_so_hash_and_build_id(self) -> None:
        result = run_bash(
            r'''
source "$1"
SIMAAI_MEMORY_RUNTIME_DEB=runtime.deb
SIMAAI_MEMORY_REQUIRED_VERSION=2.1.1
artifact_checksum_for_file() { :; }
dpkg-deb() {
  [[ "$1" == -x ]] || return 2
  mkdir -p "$3/usr/lib/aarch64-linux-gnu"
  printf 'payload' > "$3/usr/lib/aarch64-linux-gnu/libsimaaimem.so.2.1.1"
}
readelf() {
  case "$1" in
    -d) printf '%s\n' ' 0x000000000000000e (SONAME) Library soname: [libsimaaimem.so.2]' ;;
    -Ws) printf '%s\n' '37: 0000 1 FUNC GLOBAL DEFAULT 1 simaai_memory_export_dmabuf_fd' ;;
    -n) printf '%s\n' '    Build ID: feedface' ;;
    *) return 2 ;;
  esac
}
sha256sum() { printf '%064d  %s\n' 0 "$1"; }
validate_local_simaai_memory_payload
printf 'PATH=%s\nSHA=%s\nBUILD=%s\n' \
  "${SIMAAI_MEMORY_PAYLOAD_PATH}" "${SIMAAI_MEMORY_PAYLOAD_SHA256}" \
  "${SIMAAI_MEMORY_PAYLOAD_BUILD_ID}"
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PATH=/usr/lib/aarch64-linux-gnu/libsimaaimem.so.2.1.1", result.stdout)
        self.assertIn("BUILD=feedface", result.stdout)
        self.assertIn("SHA=" + "0" * 64, result.stdout)

    def test_isolated_transaction_simulates_then_installs_only_local_paths(self) -> None:
        result = run_bash(
            r'''
source "$1"
DEBS=(./memory-runtime.deb ./memory-dev.deb ./neat-runtime.deb)
collect_local_simaai_memory_debs() {
  SIMAAI_MEMORY_RUNTIME_DEB=./memory-runtime.deb
  SIMAAI_MEMORY_DEV_DEB=./memory-dev.deb
  SIMAAI_MEMORY_DEBS=("${SIMAAI_MEMORY_RUNTIME_DEB}" "${SIMAAI_MEMORY_DEV_DEB}")
}
validate_local_simaai_memory_payload() { :; }
snapshot_memory_transaction_guard_state() { :; }
verify_installed_simaai_memory_payload() { :; }
verify_memory_transaction_preservation() { :; }
run_sudo() {
  printf 'APT:'
  printf ' <%s>' "$@"
  printf '\n'
}
install_local_simaai_memory_transaction
printf 'COMPLETE=%s\n' "${SIMAAI_MEMORY_TRANSACTION_COMPLETE}"
printf 'REMAINING:'; printf ' <%s>' "${DEBS[@]}"; printf '\n'
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        apt_lines = [line for line in result.stdout.splitlines() if line.startswith("APT:")]
        self.assertEqual(len(apt_lines), 2, result.stdout)
        self.assertIn("<--simulate>", apt_lines[0])
        self.assertNotIn("<--simulate>", apt_lines[1])
        for line in apt_lines:
            self.assertIn("<--no-remove>", line)
            self.assertIn("<--reinstall>", line)
            self.assertIn("<./memory-runtime.deb>", line)
            self.assertIn("<./memory-dev.deb>", line)
            self.assertNotIn("simaai-memory-lib=", line)
            self.assertNotIn("--fix-broken", line)
            self.assertNotIn("--force-overwrite", line)
        self.assertIn("COMPLETE=1", result.stdout)
        self.assertIn("REMAINING: <./neat-runtime.deb>", result.stdout)

    def test_isolated_transaction_rejects_simulated_removal_before_real_apt(self) -> None:
        result = run_bash(
            r'''
source "$1"
DEBS=(./memory-runtime.deb ./memory-dev.deb)
collect_local_simaai_memory_debs() {
  SIMAAI_MEMORY_DEBS=(./memory-runtime.deb ./memory-dev.deb)
}
validate_local_simaai_memory_payload() { :; }
snapshot_memory_transaction_guard_state() { :; }
run_sudo() {
  case " $* " in
    *' --simulate '*)
      printf '%s\n' 'Remv simaai-palette-modalix [2.1.2]'
      ;;
    *)
      printf '%s\n' REAL_APT_CALLED
      ;;
  esac
}
install_local_simaai_memory_transaction
'''
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("planned package removal", result.stderr)
        self.assertNotIn("REAL_APT_CALLED", result.stdout + result.stderr)

    def test_preservation_check_rejects_any_preinstalled_package_loss(self) -> None:
        result = run_bash(
            r'''
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
SIMAAI_MEMORY_PREINSTALL_PACKAGES="${tmp}/before"
printf '%s\n' keep-me removed-by-resolver > "${SIMAAI_MEMORY_PREINSTALL_PACKAGES}"
verify_memory_guard_palette_and_ota() { :; }
dpkg-query() {
  [[ "$1" == -W ]] || return 2
  printf 'keep-me\tii \n'
}
run_sudo() { "$@"; }
verify_memory_transaction_preservation
'''
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("removed preinstalled packages", result.stderr)
        self.assertIn("removed-by-resolver", result.stderr)

    def test_native_restore_never_readds_memory_after_isolated_transaction(self) -> None:
        result = run_bash(
            r'''
source "$1"
SIMAAI_MEMORY_TRANSACTION_COMPLETE=1
apt_candidate_version() { printf '%s\n' 2.1.2; }
apt_exact_dependency_version() {
  case "$3" in
    libcamera|libcamera-tools) printf '%s\n' 2.1.1 ;;
    simaai-memory-lib) printf '%s\n' 2.1.1 ;;
  esac
}
exact_package_install_spec() { printf '%s=%s\n' "$1" "$2"; }
deb_package_is_present() { return 1; }
native_modalix_restore_specs specs
printf '%s\n' "${specs[@]}"
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("simaai-memory-lib", result.stdout)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "libcamera=2.1.1",
                "libcamera-tools=2.1.1",
                "simaai-gst-plugins",
                "simaai-palette-modalix=2.1.2",
            ],
        )


class DispatcherMigrationTest(unittest.TestCase):
    def test_migration_moves_unowned_global_and_backup_outside_loader_dir(self) -> None:
        result = run_bash(
            r'''
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
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MIGRATED", result.stdout)
        self.assertNotIn("Linking", result.stdout)

    def test_migration_refuses_package_owned_global_dispatcher(self) -> None:
        result = run_bash(
            r'''
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
'''
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("package-owned global dispatcher", result.stderr)

    def test_verifies_versioned_private_dispatcher_and_package_ownership(self) -> None:
        result = run_bash(
            r'''
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
global="${tmp}/loader"
runtime="${global}/neat/runtime"
mkdir -p "${runtime}"
touch "${runtime}/libneatdispatchercore.so.1.0.0"
ln -s libneatdispatchercore.so.1.0.0 "${runtime}/libneatdispatchercore.so.1"
dispatcher_global_lib_dir() { printf '%s\n' "${global}"; }
dispatcher_private_runtime_dir() { printf '%s\n' "${runtime}"; }
readelf() { printf '%s\n' ' 0x000000000000000e (SONAME) Library soname: [libneatdispatchercore.so.1]'; }
dpkg-query() { printf 'neat-runtime: %s\n' "$2"; }
verify_private_dispatcher_runtime
printf 'PRIVATE_OK\n'
'''
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PRIVATE_OK", result.stdout)
        self.assertIn("versioned package-owned dispatcher", result.stdout)



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
