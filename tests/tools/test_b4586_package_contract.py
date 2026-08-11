#!/usr/bin/env python3
"""Focused tests for the B4593 kernel and camera package contract."""

from __future__ import annotations

import subprocess
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


class B4593PackageContractTest(unittest.TestCase):
    def test_loads_reviewed_contract_from_source_manifest(self) -> None:
        result = run_bash(
            rf"""
source "$1"
NEAT_PACKAGE_MANIFEST="{ROOT / 'deps' / 'manifest.json'}"
load_b4586_platform_package_contract
printf '%s|%s|%s\n' \
  "${{B4586_KERNEL_PACKAGE_VERSION}}" \
  "${{B4586_LIBCAMERA_VERSION}}" \
  "${{B4586_MEMORY_VERSION}}"
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("6.18.3-4593|2.1.3+neat1|2.1.1-0neat4", result.stdout)

    def test_running_kernel_requires_exact_package_and_booted_build(self) -> None:
        result = run_bash(
            r"""
source "$1"
deb_package_is_installed() { [[ "$1" == linux-image-6.18.3-modalix ]]; }
deb_package_installed_version() { printf '%s\n' 6.18.3-4593; }
dpkg-query() { printf '%s\n' arm64; }
uname() { [[ "$1" == -r ]] && printf '%s\n' 6.18.3-modalix; }
cat() {
  [[ "$1" == /proc/version ]] || return 2
  printf '%s\n' 'Linux version 6.18.3-modalix (builder) #4593 SMP PREEMPT'
}
verify_b4586_running_kernel
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Verified running kernel", result.stdout)

    def test_running_kernel_rejects_installed_but_not_booted_build(self) -> None:
        result = run_bash(
            r"""
source "$1"
deb_package_is_installed() { return 0; }
deb_package_installed_version() { printf '%s\n' 6.18.3-4593; }
dpkg-query() { printf '%s\n' arm64; }
uname() { printf '%s\n' 6.18.2-modalix; }
verify_b4586_running_kernel
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not running", result.stderr)

    def test_collects_complete_libcamera_override_with_capability(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
touch "${tmp}/libcamera.deb" "${tmp}/libcamera-dev.deb" "${tmp}/libcamera-tools.deb"
DEBS=("${tmp}/libcamera.deb" "${tmp}/libcamera-dev.deb" "${tmp}/libcamera-tools.deb")
palette_required_package_version() { printf '%s\n' 2.1.3~pre4593; }
board_debian_architecture() { printf '%s\n' arm64; }
dpkg-deb() {
  [[ "$1" == -f ]] || return 2
  case "$(basename "$2"):$3" in
    libcamera.deb:Package) printf '%s\n' libcamera ;;
    libcamera-dev.deb:Package) printf '%s\n' libcamera-dev ;;
    libcamera-tools.deb:Package) printf '%s\n' libcamera-tools ;;
    *:Version) printf '%s\n' 2.1.3+neat1 ;;
    *:Architecture) printf '%s\n' arm64 ;;
    libcamera.deb:Provides)
      printf '%s\n' 'libcamera (= 2.1.3~pre4593), simaai-libcamera-dmabuf-abi (= 1)'
      ;;
    libcamera-dev.deb:Provides) printf '%s\n' 'libcamera-dev (= 2.1.3~pre4593)' ;;
    libcamera-tools.deb:Provides) printf '%s\n' 'libcamera-tools (= 2.1.3~pre4593)' ;;
    libcamera-dev.deb:Depends|libcamera-tools.deb:Depends)
      printf '%s\n' 'libc6, libcamera (= 2.1.3+neat1)'
      ;;
    *) return 2 ;;
  esac
}
collect_local_libcamera_debs
printf 'VERSION=%s\n' "${LIBCAMERA_ACTUAL_VERSION}"
printf 'COUNT=%s\n' "${#LIBCAMERA_DEBS[@]}"
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VERSION=2.1.3+neat1", result.stdout)
        self.assertIn("COUNT=3", result.stdout)

    def test_libcamera_override_rejects_missing_dmabuf_capability(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
touch "${tmp}/libcamera.deb" "${tmp}/libcamera-dev.deb" "${tmp}/libcamera-tools.deb"
DEBS=("${tmp}/libcamera.deb" "${tmp}/libcamera-dev.deb" "${tmp}/libcamera-tools.deb")
palette_required_package_version() { printf '%s\n' 2.1.3~pre4593; }
board_debian_architecture() { printf '%s\n' arm64; }
dpkg-deb() {
  [[ "$1" == -f ]] || return 2
  case "$(basename "$2"):$3" in
    libcamera.deb:Package) printf '%s\n' libcamera ;;
    libcamera-dev.deb:Package) printf '%s\n' libcamera-dev ;;
    libcamera-tools.deb:Package) printf '%s\n' libcamera-tools ;;
    *:Version) printf '%s\n' 2.1.3+neat1 ;;
    *:Architecture) printf '%s\n' arm64 ;;
    libcamera.deb:Provides) printf '%s\n' 'libcamera (= 2.1.3~pre4593)' ;;
    libcamera-dev.deb:Provides) printf '%s\n' 'libcamera-dev (= 2.1.3~pre4593)' ;;
    libcamera-tools.deb:Provides) printf '%s\n' 'libcamera-tools (= 2.1.3~pre4593)' ;;
    libcamera-dev.deb:Depends|libcamera-tools.deb:Depends)
      printf '%s\n' 'libcamera (= 2.1.3+neat1)'
      ;;
    *) return 2 ;;
  esac
}
collect_local_libcamera_debs
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("simaai-libcamera-dmabuf-abi (= 1)", result.stderr)

    def test_atomic_override_transaction_contains_both_overrides(self) -> None:
        result = run_bash(
            r"""
source "$1"
NEAT_INSTALLER_SKIP_PLATFORM_CHECK=OFF
DEBS=(./memory.deb ./memory-dev.deb ./libcamera.deb ./libcamera-dev.deb \
      ./libcamera-tools.deb ./neat-runtime.deb)
preflight_b4586_board_install() {
  PLATFORM_OVERRIDE_DEBS=(./memory.deb ./memory-dev.deb ./libcamera.deb \
    ./libcamera-dev.deb ./libcamera-tools.deb)
}
snapshot_memory_transaction_guard_state() { :; }
local_platform_overrides_require_atomic_downgrade() { return 1; }
verify_installed_simaai_memory_payload() { :; }
verify_installed_libcamera_payload() { :; }
verify_memory_transaction_preservation() { :; }
verify_simulated_preinstalled_package_changes() { :; }
run_sudo() {
  printf 'APT:'
  printf ' <%s>' "$@"
  printf '\n'
}
install_local_b4586_override_transaction
printf 'REMAINING:'; printf ' <%s>' "${DEBS[@]}"; printf '\n'
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        apt_lines = [
            line for line in result.stdout.splitlines() if line.startswith("APT:")
        ]
        self.assertEqual(len(apt_lines), 2, result.stdout)
        for line in apt_lines:
            for deb in (
                "memory.deb",
                "memory-dev.deb",
                "libcamera.deb",
                "libcamera-dev.deb",
                "libcamera-tools.deb",
            ):
                self.assertIn("<./" + deb + ">", line)
            self.assertIn("<--no-remove>", line)
        self.assertIn("REMAINING: <./neat-runtime.deb>", result.stdout)

    def test_rejects_unrelated_preinstalled_upgrade(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
simulation="${tmp}/simulation.log"
deb="${tmp}/libcamera.deb"
touch "${deb}"
printf '%s\n' \
  'Inst libcamera [2.1.3~pre4593] (2.1.3+neat1 local [arm64])' \
  'Inst unrelated-runtime [1.0] (1.1 repository [arm64])' > "${simulation}"
dpkg-deb() { printf '%s\n' libcamera; }
verify_simulated_preinstalled_package_changes "${simulation}" "${deb}"
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preinstalled unrelated package unrelated-runtime", result.stderr)

    def test_preflight_failure_happens_before_service_stop(self) -> None:
        result = run_bash(
            r"""
source "$1"
DEBS=(./neat-runtime.deb)
prepare_debs_for_board_install() { :; }
preflight_b4586_board_install() { return 1; }
stop_board_runtime_before_install() { printf '%s\n' SERVICE_STOPPED; }
install_debs_on_board
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("SERVICE_STOPPED", result.stdout + result.stderr)
        self.assertIn("no board services or packages were changed", result.stderr)

    def test_complete_apt_simulation_failure_happens_before_service_stop(self) -> None:
        result = run_bash(
            r"""
source "$1"
DEBS=(./neat-runtime.deb)
prepare_debs_for_board_install() { :; }
preflight_b4586_board_install() { :; }
refresh_apt_metadata_for_board_install() { :; }
apt_package_database_is_healthy() { return 0; }
simulate_complete_board_transaction_preflight() { return 1; }
stop_board_runtime_before_install() { printf '%s\n' SERVICE_STOPPED; }
install_debs_on_board
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("SERVICE_STOPPED", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
