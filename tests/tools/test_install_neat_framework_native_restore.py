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


if __name__ == "__main__":
    if "--installer" in sys.argv:
        index = sys.argv.index("--installer")
        try:
            INSTALLER = Path(sys.argv[index + 1]).resolve()
        except IndexError as exc:
            raise SystemExit("--installer requires a path") from exc
        del sys.argv[index : index + 2]
    unittest.main()
