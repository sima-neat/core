"""Focused tests for native Modalix package handling in the installer."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "tools" / "install_neat_framework.sh"


def run_bash(script: str) -> subprocess.CompletedProcess[str]:
    """Resolve INSTALLER at call time so --installer still applies."""
    return subprocess.run(
        ["bash", "-c", script, "bash", str(INSTALLER)],
        check=False,
        text=True,
        capture_output=True,
    )


def write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


class CustomerSdkDependenciesTest(unittest.TestCase):
    def test_runtime_package_does_not_request_implementation_dev_packages(self) -> None:
        result = run_bash(r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
cd "${tmp}"
touch sima-lmm-core.deb
apt-get() { printf 'DOWNLOAD %s\n' "$*"; return 1; }
ensure_sima_lmm_sysroot_deps "${tmp}/sysroot"
""")
        self.assertNotEqual(result.returncode, 0)
        requested = next(line for line in result.stdout.splitlines() if line.startswith("DOWNLOAD"))
        self.assertIn("nlohmann-json3-dev", requested)
        self.assertIn("libcpp-httplib0.18:arm64", requested)
        self.assertIn("libavcodec61:arm64", requested)
        for package in ("libeigen3-dev", "libfmt-dev", "libspdlog-dev", "libbrotli-dev",
                        "libcpp-httplib-dev", "libfftw3-dev", "libavcodec-dev"):
            self.assertNotIn(package, requested)

    def test_core_runtime_dependencies_do_not_need_sima_lmm(self) -> None:
        result = run_bash(r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
cd "${tmp}"
apt-get() { printf 'DOWNLOAD %s\n' "$*"; return 1; }
ensure_sima_lmm_sysroot_deps "${tmp}/sysroot"
""")
        self.assertNotEqual(result.returncode, 0)
        requested = next(line for line in result.stdout.splitlines() if line.startswith("DOWNLOAD"))
        self.assertIn("libcpp-httplib0.18:arm64", requested)
        self.assertIn("libbrotli1:arm64", requested)
        self.assertNotIn("libfmt10:arm64", requested)

class ModalixBoardRouteTest(unittest.TestCase):
    def test_board_route_preserves_later_install_failures(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
INSTALLER_TMP_DIRS=("${tmp}")
ENV_MODE=modalix-board
install_board_sentinel() { printf 'SENTINEL\n'; }
install_python_environment() { printf 'PYNEAT\n'; }
install_debs_on_board() { printf 'DEBS\n'; }
install_agent_skills_for_current_user() { printf 'SKILLS\n'; return 23; }
install_for_environment
"""
        )

        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["SENTINEL", "PYNEAT", "DEBS", "SKILLS"],
        )


class ModalixSentinelInstallTest(unittest.TestCase):
    """install_sentinel.sh against fake systemctl, curl, and sima-cli commands."""

    LATEST = "766066b74f49"
    FAKE_COMMANDS = {
        "systemctl": '[[ -e "${STATE}/version" ]]',
        "curl": '[[ -e "${STATE}/version" ]] && printf \'{\n  "version": "%s"\n}\n\' "$(<"${STATE}/version")"',
        "sleep": ":",
    }

    def run_install(
        self, *, running: str | None, latest: str | None = LATEST, starts: bool = True,
        install_ok: bool = True, sima_cli: str | None = "path"
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        state = tmp / "state"
        bin_dir = tmp / "bin"
        home = tmp / "home"
        for directory in (state, bin_dir, home):
            directory.mkdir()
        if running:
            (state / "version").write_text(running)
        if latest:
            (state / "latest").write_text(latest)

        commands = dict(self.FAKE_COMMANDS)
        sima_cli_script = (
            'if [[ " $* " == *" --json "* ]]; then '
            '[[ -e "${STATE}/latest" ]] || { echo "Error: GET latest.tag failed" >&2; exit 1; }; '
            'printf \'{\\n  "resolved_spec": "%s"\\n}\\n\' "$(<"${STATE}/latest")"; exit 0; fi; '
            'printf \'SIMA-CLI %s\\n\' "$*"; printf \'%s\\n\' "$*" >"${STATE}/args"; '
            'printf \'SIMA_CLI=%s\\n\' "${SIMA_CLI}"; '
            + ('printf \'main:%s\' "${3#sentinel@}" >"${STATE}/version"; ' if starts else "")
            + f"exit {0 if install_ok else 1}"
        )
        if sima_cli == "path":
            commands["sima-cli"] = sima_cli_script
        elif sima_cli == "venv":
            venv_bin = home / ".sima-cli" / ".venv" / "bin"
            venv_bin.mkdir(parents=True)
            write_executable(venv_bin / "sima-cli", sima_cli_script)
        for name, body in commands.items():
            write_executable(bin_dir / name, body)

        result = subprocess.run(
            ["bash", str(INSTALLER.parent / "install_sentinel.sh")],
            check=False,
            text=True,
            capture_output=True,
            env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(home), "STATE": str(state)},
        )
        return result, state

    def assert_installed_latest(self, result: subprocess.CompletedProcess[str], state: Path) -> None:
        self.assertEqual(result.returncode, 0, result.stderr)
        args = (state / "args").read_text().split()
        self.assertEqual(args[:5], ["neat", "install", f"sentinel@{self.LATEST}", "--prod", "--force"])
        self.assertEqual(args[5], "--install-dir")
        self.assertFalse(Path(args[6]).exists(), "the download directory must be removed")
        self.assertIn(f"Sentinel main:{self.LATEST} is running.", result.stdout)

    def test_latest_sentinel_is_kept(self) -> None:
        result, _ = self.run_install(running=f"main:{self.LATEST}")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("SIMA-CLI", result.stdout)
        self.assertIn(f"Sentinel main:{self.LATEST} is up to date.", result.stdout)

    def test_older_sentinel_is_upgraded(self) -> None:
        self.assert_installed_latest(*self.run_install(running="main:2b0360551b5f"))

    def test_branch_build_is_replaced(self) -> None:
        self.assert_installed_latest(*self.run_install(running="feat-telemetry:0123456789ab"))

    def test_missing_sentinel_is_installed(self) -> None:
        self.assert_installed_latest(*self.run_install(running=None))

    def test_sima_cli_outside_path_is_found(self) -> None:
        result, _ = self.run_install(running=None, sima_cli="venv")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIMA-CLI neat install sentinel@", result.stdout)
        self.assertRegex(result.stdout, r"SIMA_CLI=\S+/\.sima-cli/\.venv/bin/sima-cli")

    def test_install_error_with_latest_running_warns(self) -> None:
        result, _ = self.run_install(running=None, install_ok=False)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("WARNING: 'sima-cli neat install sentinel' reported a failure", result.stdout)

    def test_sentinel_left_on_older_build_fails(self) -> None:
        result, _ = self.run_install(running="main:2b0360551b5f", starts=False, install_ok=False)

        self.assertEqual(result.returncode, 1)
        self.assertIn(f"Sentinel main:{self.LATEST} is not running after installation", result.stderr)

    def test_unresolved_build_keeps_running_sentinel(self) -> None:
        result, _ = self.run_install(running="main:2b0360551b5f", latest=None)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("SIMA-CLI", result.stdout)
        self.assertIn("keeping Sentinel main:2b0360551b5f", result.stdout)

    def test_unresolved_build_without_sentinel_fails(self) -> None:
        result, _ = self.run_install(running=None, latest=None)

        self.assertEqual(result.returncode, 1)
        self.assertNotIn("SIMA-CLI", result.stdout)
        self.assertIn("Could not resolve the latest Sentinel build", result.stderr)

    def test_missing_sima_cli_fails(self) -> None:
        result, _ = self.run_install(running=None, sima_cli=None)

        self.assertEqual(result.returncode, 1)
        self.assertNotIn("SIMA-CLI", result.stdout)
        self.assertIn("sima-cli is required to install Sentinel", result.stderr)

    @staticmethod
    def run_board_sentinel(required: str) -> subprocess.CompletedProcess[str]:
        """Run install_board_sentinel with a failing sibling script, then with none."""
        return run_bash(
            rf"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${{tmp}}"' EXIT
NEAT_INSTALLER_SENTINEL_REQUIRED={required}
SENTINEL_INSTALLER_PATH="${{tmp}}/install_sentinel.sh"
printf 'echo SENTINEL-SCRIPT; exit 7\n' >"${{SENTINEL_INSTALLER_PATH}}"
(install_board_sentinel; printf 'CONTINUED\n') || printf 'EXIT %s\n' "$?"
rm "${{SENTINEL_INSTALLER_PATH}}"
(install_board_sentinel; printf 'CONTINUED\n') || printf 'EXIT %s\n' "$?"
"""
        )

    def test_sentinel_failure_warns_and_continues_by_default(self) -> None:
        result = self.run_board_sentinel("OFF")

        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], "SENTINEL-SCRIPT")
        self.assertEqual(lines.count("CONTINUED"), 2, result.stdout)
        self.assertEqual(
            sum("WARNING: Sentinel is not installed" in line for line in lines), 2, result.stdout
        )
        self.assertIn("sima-cli neat install sentinel", result.stdout)
        self.assertIn("install_sentinel.sh; it ships with this installer", result.stderr)

    def test_required_sentinel_failure_exits(self) -> None:
        result = self.run_board_sentinel("ON")

        self.assertEqual(result.stdout.splitlines(), ["SENTINEL-SCRIPT", "EXIT 1", "EXIT 1"])
        self.assertIn("NEAT_INSTALLER_SENTINEL_REQUIRED=ON", result.stderr)
        self.assertIn("install_sentinel.sh; it ships with this installer", result.stderr)

    def test_sentinel_installer_receives_the_sudo_password(self) -> None:
        result = run_bash(
            r"""
unset SUDO_PASSWORD DEVKIT_PASSWORD DEFAULT_SUDO_PASSWORD
DEVKIT_PASSWORD=devkit-pw
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
SENTINEL_INSTALLER_PATH="${tmp}/install_sentinel.sh"
printf 'printf "SENTINEL SUDO_PASSWORD=%%s\n" "${SUDO_PASSWORD-unset}"\n' \
  >"${SENTINEL_INSTALLER_PATH}"
install_board_sentinel
SUDO_PASSWORD=""
install_board_sentinel
"""
        )

        self.assertEqual(
            result.stdout.splitlines(),
            ["SENTINEL SUDO_PASSWORD=devkit-pw", "SENTINEL SUDO_PASSWORD="],
            result.stderr,
        )

    def test_sentinel_is_optional_unless_required(self) -> None:
        result = run_bash('source "$1"; printf "%s\n" "${NEAT_INSTALLER_SENTINEL_REQUIRED}"')

        self.assertEqual(result.stdout.strip(), "OFF", result.stderr)


class ModalixPythonVenvSupportTest(unittest.TestCase):
    def test_dangling_interpreter_is_repaired_without_removing_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / "pyneat"
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
            marker = venv / "lib" / "preserved-package"
            marker.write_text("existing package")
            for alias in (venv / "bin").glob("python*"):
                if alias.is_symlink():
                    alias.unlink()
                    alias.symlink_to("/missing/previous-sdk/python3")
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
            self.assertFalse((venv / "bin" / "python").exists())
            result = run_bash(f'''
source "$1"
prepare_python_venv '{venv}'
"{venv}/bin/python" -c 'import sys; assert sys.prefix != sys.base_prefix'
''')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Removing dangling virtualenv interpreter link", result.stdout)
            self.assertEqual(marker.read_text(), "existing package")

    def test_working_interpreter_links_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / "pyneat"
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
            interpreter = venv / "bin" / "python"
            before = interpreter.lstat().st_ino
            result = run_bash(f'source "$1"; prepare_python_venv "{venv}"')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Removing dangling", result.stdout)
            self.assertEqual(interpreter.lstat().st_ino, before)

    @staticmethod
    def run_install(
        *, ensurepip_available: bool, apt_install_ok: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return run_bash(
            rf"""
source "$1"
tmp="$(mktemp -d)"
INSTALLER_TMP_DIRS=("${{tmp}}")
{'touch "${tmp}/ensurepip"' if ensurepip_available else ''}
python3() {{ [[ -e "${{tmp}}/ensurepip" ]]; }}
refresh_apt_metadata_for_board_install() {{ printf 'REFRESH\n'; }}
run_sudo() {{
  printf 'RUN: %s\n' "$*"
  {'touch "${tmp}/ensurepip"' if apt_install_ok else 'return 100'}
}}
# First step after the check: reaching it means the installer continued.
remove_stale_global_sima_lmm_pip_install() {{ printf 'CONTINUED\n'; exit 0; }}
install_python_environment
"""
        )

    def test_available_ensurepip_skips_apt(self) -> None:
        result = self.run_install(ensurepip_available=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["CONTINUED"])

    def test_missing_ensurepip_installs_python3_venv_and_continues(self) -> None:
        result = self.run_install(ensurepip_available=False)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [line for line in result.stdout.splitlines() if not line.startswith("[")],
            ["REFRESH", "RUN: apt-get install -y python3-venv", "CONTINUED"],
        )

    def test_failed_python3_venv_install_stops_installer(self) -> None:
        result = self.run_install(ensurepip_available=False, apt_install_ok=False)

        self.assertEqual(result.returncode, 1)
        self.assertNotIn("CONTINUED", result.stdout)
        self.assertIn("sudo apt-get install -y python3-venv", result.stderr)


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
dpkg-deb() {
  case "$3" in
    Package) basename "$2" .deb ;;
    Version) echo 1.2.3 ;;
    Architecture) echo arm64 ;;
    *) return 1 ;;
  esac
}
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
        for name in ("sima-neat", "sima-neat-dev", "neat-runtime"):
            self.assertIn(f"<{name}:arm64=1.2.3>", transaction)

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
dpkg-deb() {
  case "$3" in
    Package) basename "$2" .deb ;;
    Version) echo 1.2.3 ;;
    Architecture) echo arm64 ;;
    *) return 1 ;;
  esac
}
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


class ForceInstallTest(unittest.TestCase):
    def run_activation_fixture(
        self, script: str, exit_code: int = 0, *, selected: bool = True,
        invalid: str = "", package: str = "neat-ev74-firmware", required: bool | None = None,
        unrelated: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        # Redirect only the fixed firmware path in a temporary installer copy;
        # never create or execute firmware under the host's /usr/libexec.
        # The selected archive is real; only the installed dpkg database is faked.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            firmware = root / "firmware.sh"
            write_executable(
                firmware,
                'printf "FIRMWARE_FORCE=%s ARG=%s\\n" "${FORCE_INSTALL-unset}" "$1"\n'
                f"exit {exit_code}",
            )
            payload = root / "package"
            control = payload / "DEBIAN" / "control"
            control.parent.mkdir(parents=True)
            package_architecture = "arm64" if required else "all"
            control.write_text(
                f"Package: {package}\nVersion: 0.6.0+fixture\nArchitecture: {package_architecture}\n"
                "Maintainer: Test <test@example.invalid>\nDescription: inert activation fixture\n"
            )
            packaged_helper = payload / firmware.relative_to("/")
            packaged_helper.parent.mkdir(parents=True)
            shutil.copy2(firmware, packaged_helper)
            if invalid == "packaged-symlink":
                packaged_helper.unlink()
                packaged_helper.symlink_to(firmware)
            elif invalid == "packaged-mode":
                packaged_helper.chmod(0o644)
            elif invalid == "packaged-directory":
                packaged_helper.unlink()
                packaged_helper.mkdir()
            elif invalid == "unselected-payload":
                packaged_helper.unlink()
            archive = root / "firmware.deb"
            subprocess.run(["dpkg-deb", "--build", "--root-owner-group", str(payload), str(archive)],
                           check=True, capture_output=True, text=True)
            required_identity = {
                "Package": package, "Version": "0.6.0+fixture",
                "Architecture": package_architecture,
                "deb_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
            if invalid == "required-archive":
                (payload / "changed-payload").write_text("changed after admission")
                subprocess.run(["dpkg-deb", "--build", "--root-owner-group", str(payload), str(archive)],
                               check=True, capture_output=True, text=True)
            for field in ("Package", "Version", "Architecture", "deb_sha256"):
                if invalid == "required-" + field:
                    required_identity[field] = "different"
            if invalid == "archive":
                archive.write_bytes(b"not a Debian archive")
            if invalid == "missing":
                firmware.unlink()
            elif invalid == "symlink":
                target = root / "other.sh"
                firmware.rename(target)
                firmware.symlink_to(target)
            elif invalid == "content":
                firmware.write_text(firmware.read_text() + "# not selected bytes\n")
            elif invalid == "mode":
                firmware.chmod(0o700)
            version = "0.5.0" if invalid == "version" else "0.6.0+fixture"
            architecture = ("all" if required else "arm64") if invalid == "architecture" else package_architecture
            status = "iU " if invalid == "status" else "ii "
            owner = "unrelated-package" if invalid == "owner" else package
            debs = f'"{archive}"' if selected else ""
            if invalid == "duplicate":
                debs += f' "{archive}"'
            if unrelated:
                other_payload = root / "other-package"
                other_control = other_payload / "DEBIAN" / "control"
                other_control.parent.mkdir(parents=True)
                other_control.write_text(
                    "Package: unrelated-product\nVersion: 1\nArchitecture: all\n"
                    "Maintainer: Test <test@example.invalid>\nDescription: unrelated fixture\n"
                )
                (other_payload / "unrelated-payload").write_text("not firmware")
                other_archive = root / "other.deb"
                subprocess.run(["dpkg-deb", "--build", "--root-owner-group", str(other_payload), str(other_archive)],
                               check=True, capture_output=True, text=True)
                debs += f' "{other_archive}"'
            fixture = f"""DEBS=({debs})
dpkg-query() {{
  case "$1" in
    -W) printf '%s\\t%s\\t%s\\t%s' '{package}' '{version}' '{architecture}' '{status}' ;;
    -S) printf '%s: %s\\n' '{owner}' '{firmware}' ;;
    *) return 99 ;;
  esac
}}
"""
            if required is not None:
                admission_dir = root / "admission"
                admission_dir.mkdir()
                (admission_dir / "before.json").write_text(json.dumps({
                    "split_cohort": True,
                    "firmware": {"required": required, "debian_package": required_identity if required else None},
                }))
                fixture += f'BOARD_PLATFORM_ADMISSION_DIR="{admission_dir}"\n'
            if unrelated:
                fixture += f'''dpkg-deb() {{
  if [[ "$2" == "{other_archive}" && ( "$1" == --fsys-tarfile || "$1" == -x ) ]]; then
    echo UNEXPECTED_UNRELATED_PAYLOAD >&2
    return 73
  fi
  command dpkg-deb "$@"
}}
'''
            if invalid == "extraction":
                fixture += 'dpkg-deb() { [[ "$1" != -x ]] || return 72; command dpkg-deb "$@"; }\n'
            installer = root / "installer.sh"
            source = INSTALLER.read_text()
            fixed_path = "/usr/libexec/sima-neat-firmware/install.sh"
            self.assertIn(fixed_path, source)
            # Likewise keep the boot link out of the host's /lib/firmware.
            boot_link_path = "/lib/firmware/updates/modalix-cvu-fw"
            self.assertIn(boot_link_path, source)
            boot_link = root / "firmware-updates" / "modalix-cvu-fw"
            fixture += f'BOOT_LINK="{boot_link}"\n'
            guard = 'if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then'
            self.assertEqual(source.count(guard), 1)
            installer.write_text(source.replace(fixed_path, str(firmware))
                                 .replace(boot_link_path, str(boot_link))
                                 .replace(guard, fixture + guard))
            return subprocess.run(
                ["bash", "-c", script, "bash", str(installer)],
                check=False,
                text=True,
                capture_output=True,
            )

    def test_ambient_firmware_without_selected_package_is_not_activated(self) -> None:
        result = self.run_activation_fixture(r"""
source "$1"
FORCE_INSTALL=1
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON
run_sudo() { printf 'UNEXPECTED_ACTIVATION\n'; return 99; }
activate_board_runtime_after_install
""", selected=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("EV74 firmware activation skipped", result.stdout)
        self.assertNotIn("UNEXPECTED_ACTIVATION", result.stdout)

    def test_selected_firmware_must_match_installed_identity_owner_and_payload(self) -> None:
        for invalid in ("version", "architecture", "status", "owner", "missing", "symlink",
                        "content", "mode", "packaged-symlink", "packaged-mode", "packaged-directory",
                        "duplicate", "archive", "extraction"):
            with self.subTest(invalid=invalid):
                result = self.run_activation_fixture(r"""
source "$1"
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON
FORCE_INSTALL=1
run_sudo() { printf 'UNEXPECTED_ACTIVATION\n'; }
activate_board_runtime_after_install
""", invalid=invalid)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("UNEXPECTED_ACTIVATION", result.stdout)

    def test_other_selected_package_does_not_activate_ambient_firmware(self) -> None:
        result = self.run_activation_fixture(r"""
source "$1"
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON
FORCE_INSTALL=1
run_sudo() { printf 'UNEXPECTED_ACTIVATION\n'; return 99; }
activate_board_runtime_after_install
""", package="other-product", invalid="unselected-payload")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("EV74 firmware activation skipped", result.stdout)
        self.assertNotIn("UNEXPECTED_ACTIVATION", result.stdout)

    def test_required_native_firmware_activates_only_the_admitted_package(self) -> None:
        for package in ("simaai-kernels-firmware", "renamed-firmware-provider"):
            with self.subTest(package=package):
                result = self.run_activation_fixture(r"""
source "$1"
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON
run_sudo() { env -i PATH="$PATH" "$@"; }
activate_board_runtime_after_install
printf 'BOOT_LINK_TARGET=%s\n' "$(readlink "${BOOT_LINK}")"
""", package=package, required=True, unrelated=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("FIRMWARE_FORCE=0 ARG=--activate", result.stdout)
                self.assertNotIn("UNEXPECTED_UNRELATED_PAYLOAD", result.stderr)
                self.assertIn("BOOT_LINK_TARGET=/lib/firmware/sima-neat-native-cvu-fw", result.stdout)

    def test_non_ev_split_cohort_skips_firmware_discovery(self) -> None:
        result = self.run_activation_fixture(r"""
source "$1"
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON
dpkg-deb() { echo UNEXPECTED_DEB_INSPECTION >&2; return 99; }
run_sudo() { printf 'UNEXPECTED_ACTIVATION\n'; }
activate_board_runtime_after_install
""", required=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("not required by the selected Scheduler cohort", result.stdout)
        self.assertNotIn("UNEXPECTED_", result.stdout + result.stderr)

    def test_legacy_firmware_discovery_still_inspects_selected_payloads(self) -> None:
        result = self.run_activation_fixture(r"""
source "$1"
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON
run_sudo() { printf 'UNEXPECTED_ACTIVATION\n'; }
activate_board_runtime_after_install
""", unrelated=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("UNEXPECTED_UNRELATED_PAYLOAD", result.stderr)
        self.assertIn("Cannot inspect selected DEB firmware payload", result.stderr)
        self.assertNotIn("UNEXPECTED_ACTIVATION", result.stdout)

    def test_required_native_firmware_cannot_be_missing_or_substituted(self) -> None:
        for invalid in ("unselected-payload", "required-Package", "required-Version",
                        "required-Architecture", "required-deb_sha256", "required-archive", "duplicate"):
            with self.subTest(invalid=invalid):
                result = self.run_activation_fixture(r"""
source "$1"
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON
FORCE_INSTALL=1
NEAT_INSTALLER_SKIP_PLATFORM_CHECK=ON
run_sudo() { printf 'UNEXPECTED_ACTIVATION\n'; }
activate_board_runtime_after_install
""", package="simaai-kernels-firmware", required=True, invalid=invalid)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("UNEXPECTED_ACTIVATION", result.stdout)
        result = self.run_activation_fixture(r"""
source "$1"
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON
run_sudo() { printf 'UNEXPECTED_ACTIVATION\n'; }
activate_board_runtime_after_install
""", required=True, selected=False)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Required native EV74 firmware", result.stderr)
        self.assertNotIn("UNEXPECTED_ACTIVATION", result.stdout)

    def test_selected_payload_owner_can_be_renamed(self) -> None:
        result = self.run_activation_fixture(r"""
source "$1"
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON
run_sudo() { env -i PATH="$PATH" "$@"; }
activate_board_runtime_after_install
""", package="renamed-firmware-provider")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FIRMWARE_FORCE=0 ARG=--activate", result.stdout)

    def test_activation_normalizes_force_after_privilege_environment_filtering(self) -> None:
        for sanitized in (False, True):
            for value in (None, "", "0", "1", "true", "ON"):
                with self.subTest(sanitized=sanitized, value=value):
                    force_setup = "unset FORCE_INSTALL" if value is None else f"FORCE_INSTALL='{value}'"
                    # The sanitized stand-in models sudo env_reset. FORCE_INSTALL
                    # is deliberately not exported: argv must carry the override.
                    privilege_boundary = (
                        'run_sudo() { env -i PATH="$PATH" "$@"; }'
                        if sanitized else 'run_sudo() { "$@"; }'
                    )
                    result = self.run_activation_fixture(
                        'source "$1"\nunset FORCE_INSTALL\n'
                        f"{force_setup}\n{privilege_boundary}\n"
                        "NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON\n"
                        "NEAT_INSTALLER_SKIP_PLATFORM_CHECK=ON\n"
                        "activate_board_runtime_after_install\n"
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = "1" if value == "1" else "0"
                    self.assertIn(f"FIRMWARE_FORCE={expected} ARG=--activate", result.stdout)

    def test_force_does_not_enable_disabled_activation(self) -> None:
        result = self.run_activation_fixture(r"""
source "$1"
FORCE_INSTALL=1
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=OFF
run_sudo() { printf 'UNEXPECTED_PRIVILEGE_CALL\n'; return 99; }
activate_board_runtime_after_install
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("EV74 firmware activation skipped", result.stdout)
        self.assertNotIn("FIRMWARE_FORCE=", result.stdout)
        self.assertNotIn("UNEXPECTED_PRIVILEGE_CALL", result.stdout)

    def test_activation_failure_stops_board_completion(self) -> None:
        result = self.run_activation_fixture(r"""
source "$1"
FORCE_INSTALL=1
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON
run_sudo() { env -i PATH="$PATH" "$@"; }
board_runtime_is_legacy() { return 1; }
repair_global_sima_neat_lib_links() { :; }
verify_global_sima_neat_lib_links() { :; }
verify_canonical_palette_installation() { :; }
restart_board_codec_services() { printf 'UNEXPECTED_RESTART\n'; }
verify_board_codec_services() { printf 'UNEXPECTED_VERIFY\n'; }
complete_board_install_after_packages
printf 'UNEXPECTED_SUCCESS\n'
""", exit_code=23)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertIn("FIRMWARE_FORCE=1 ARG=--activate", result.stdout)
        self.assertNotIn("UNEXPECTED_", result.stdout)

    def test_activation_failure_stops_conditional_board_completion(self) -> None:
        result = self.run_activation_fixture(r"""
source "$1"
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON
run_sudo() { env -i PATH="$PATH" "$@"; }
board_runtime_is_legacy() { return 1; }
repair_global_sima_neat_lib_links() { :; }
verify_global_sima_neat_lib_links() { :; }
verify_canonical_palette_installation() { :; }
restart_board_codec_services() { printf 'UNEXPECTED_RESTART\n'; }
verify_board_codec_services() { printf 'UNEXPECTED_VERIFY\n'; }
if complete_board_install_after_packages; then
  printf 'UNEXPECTED_SUCCESS\n'
  exit 0
else
  status=$?
  [[ ! -e "${BOOT_LINK}" && ! -L "${BOOT_LINK}" ]] || printf 'UNEXPECTED_BOOT_LINK\n'
  exit "${status}"
fi
""", exit_code=23)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertNotIn("UNEXPECTED_", result.stdout)

    def test_paired_force_reaches_activation_in_clean_remote_environment(self) -> None:
        for value in ("0", "1"):
            with self.subTest(value=value):
                result = self.run_activation_fixture(
                    r"""
source "$1"
NEAT_INSTALLER_SKIP_DEVKIT_SYNC=OFF
DEVKIT_SYNC_DEVKIT_IP=example.invalid
collect_cached_devkit_deploy_files() { CACHED_DEPLOY_FILES=(fake.deb); }
run_scp() { :; }
run_ssh() {
  if [[ "$1" == -t ]]; then
    eval "set -- ${3#bash -lc }"
    # Execute the actual export from the SSH payload in a clean shell, then
    # invoke the real activation function with a sanitized privilege boundary.
    local force_export
    force_export="$(printf '%s\n' "$1" | grep '^export FORCE_INSTALL=')"
    env -i PATH="$PATH" bash -c "${force_export}
source \"\$1\"
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON
run_sudo() { env -i PATH=\"\$PATH\" \"\$@\"; }
activate_board_runtime_after_install
" bash "${INSTALLER_SCRIPT_PATH}"
  fi
}
"""
                    + f"\nFORCE_INSTALL={value}\ndeploy_artifacts_to_paired_devkit_if_configured\n"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"FIRMWARE_FORCE={value} ARG=--activate", result.stdout)

    def test_only_one_bypasses_platform_mismatch(self) -> None:
        for mode in ("elxr-sdk", "ros2-sdk", "modalix-board"):
            for value in (None, "", "0", "1", "true", "ON"):
                with self.subTest(mode=mode, value=value):
                    force_setup = "unset FORCE_INSTALL" if value is None else f"export FORCE_INSTALL='{value}'"
                    result = run_bash(
                        r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
printf '%s\n' '{"platform-version":"3.0.0"}' > "${tmp}/manifest.json"
printf '%s\n' 'Platform Version = 2.1.3' 'SDK Version = 2.1.3' > "${tmp}/sdk-release"
printf '%s\n' 'MACHINE = modalix' 'DISTRO_VERSION = 2.1.3' > "${tmp}/buildinfo"
NEAT_PACKAGE_MANIFEST="${tmp}/manifest.json"
ELXR_SDK_RELEASE_FILE="${tmp}/sdk-release"
NEAT_BUILDINFO_FILE="${tmp}/buildinfo"
NEAT_INSTALLER_SKIP_PLATFORM_CHECK=OFF
"""
                        + f"\nENV_MODE={mode}\n{force_setup}\nensure_platform_compatible\n"
                    )
                    if value == "1":
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("FORCE_INSTALL=1; skipping", result.stdout)
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("Incompatible platform version", result.stderr)

    def test_force_is_forwarded_to_paired_devkit(self) -> None:
        for value in ("0", "1"):
            with self.subTest(value=value):
                result = run_bash(
                    r"""
source "$1"
NEAT_INSTALLER_SKIP_DEVKIT_SYNC=OFF
DEVKIT_SYNC_DEVKIT_IP=example.invalid
collect_cached_devkit_deploy_files() { CACHED_DEPLOY_FILES=(fake.deb); }
run_scp() { :; }
run_ssh() {
  if [[ "$1" == -t ]]; then
    [[ "$3" == 'bash -lc '* ]]
    # Decode the quoted remote payload without executing an installation.
    eval "set -- ${3#bash -lc }"
    printf '%s\n' "$1"
  fi
}
"""
                    + f"\nexport FORCE_INSTALL={value}\ndeploy_artifacts_to_paired_devkit_if_configured\n"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"export FORCE_INSTALL={value}", result.stdout)


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
replacement="${tmp}/replacement-runtime.deb"
simulation="${tmp}/simulation.log"
touch "${replacement}"
printf '%s\n' 'Remv platform-runtime [2.1.3~pre4678]' > "${simulation}"
dpkg-query() {
  printf '%s\n' '2.1.3~pre4678'
}
dpkg-deb() {
  [[ "$1" == -f ]] || return 2
  case "$3" in
    Provides) printf '%s\n' 'platform-runtime (= 2.1.3~pre4678)' ;;
    Replaces) printf '%s\n' 'platform-runtime' ;;
    Conflicts) printf '%s\n' 'platform-runtime' ;;
    *) return 2 ;;
  esac
}
verify_simulated_package_removals "${simulation}" "${replacement}"
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Verified platform package replacements", result.stdout)
        self.assertIn("platform-runtime=2.1.3~pre4678", result.stdout)

    def test_board_transaction_rejects_non_exact_replacement(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
replacement="${tmp}/replacement-runtime.deb"
simulation="${tmp}/simulation.log"
touch "${replacement}"
printf '%s\n' 'Remv platform-runtime [2.1.3~pre4678]' > "${simulation}"
dpkg-query() {
  printf '%s\n' '2.1.3~pre4678'
}
dpkg-deb() {
  [[ "$1" == -f ]] || return 2
  case "$3" in
    Provides) printf '%s\n' 'platform-runtime (= 2.1.3)' ;;
    Replaces) printf '%s\n' 'platform-runtime' ;;
    Conflicts) printf '%s\n' 'platform-runtime' ;;
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

    def test_board_transaction_accepts_retired_neat_package_removals(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
simulation="${tmp}/simulation.log"
printf '%s\n' \
  'Remv neat-libcamera [2.1.1~pre3348]' \
  'Remv neat-libcamera-dev [2.1.1~pre3348]' \
  'Remv neat-libcamera-tools [2.1.1~pre3348]' \
  'Remv neat-internals-dev [0.5.0+3.0.0-prep.6223db77f5b8]' \
  'Remv sima-lmm-dev [0.4.0+3.0.0-prep.0762b12269dd]' > "${simulation}"
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


class FirmwareRetirementTest(unittest.TestCase):
    def test_retirement_requires_admitted_archive_and_matching_versioned_relations(self):
        cases = (
            ("conflicts", True), ("breaks", True), ("narrower-bound", True),
            ("missing-replaces", False), ("missing-conflict", False),
            ("unbounded-replaces", False), ("unbounded-conflict", False),
            ("future-version", False), ("mismatched-bound", False),
            ("unrelated-removal", False), ("non-ev", False),
            ("missing-admission", False), ("wrong-identity", False),
            ("changed-archive", False), ("unselected-archive", False),
        )
        for case, accepted in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload = root / "payload"
                (payload / "DEBIAN").mkdir(parents=True)
                bound = "0.6.1~" if case == "narrower-bound" else "0.7.0"
                if case == "mismatched-bound":
                    bound = "0.6.0"
                relation = f"neat-ev74-firmware (<< {bound})"
                fields = {
                    "Package": "simaai-kernels-firmware", "Version": "0.1.0+fixture",
                    "Architecture": "arm64", "Maintainer": "Test <test@example.invalid>",
                    "Description": "inert firmware retirement fixture",
                    "Replaces": relation, "Breaks" if case == "breaks" else "Conflicts": relation,
                }
                if case == "missing-replaces":
                    del fields["Replaces"]
                elif case == "missing-conflict":
                    del fields["Conflicts"]
                elif case == "unbounded-replaces":
                    fields["Replaces"] = "neat-ev74-firmware"
                elif case == "unbounded-conflict":
                    fields["Conflicts"] = "neat-ev74-firmware"
                (payload / "DEBIAN/control").write_text(
                    "".join(f"{key}: {value}\n" for key, value in fields.items()))
                archive = root / "firmware.deb"
                subprocess.run(["dpkg-deb", "--build", "--root-owner-group", str(payload), str(archive)],
                               check=True, capture_output=True)
                control = {key: fields[key] for key in ("Package", "Version", "Architecture")}
                control["deb_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
                if case == "wrong-identity":
                    control["Package"] = "other-firmware"
                if case == "changed-archive":
                    (payload / "changed").write_text("different selected bytes")
                    subprocess.run(["dpkg-deb", "--build", "--root-owner-group", str(payload), str(archive)],
                                   check=True, capture_output=True)
                (root / "before.json").write_text(json.dumps({
                    "split_cohort": True,
                    "firmware": {"required": case != "non-ev",
                                 "debian_package": None if case == "non-ev" else control},
                }))
                removed = "unrelated-package" if case == "unrelated-removal" else "neat-ev74-firmware"
                version = "0.7.0" if case == "future-version" else "0.6.0+develop.ce6b01ccdc94"
                (root / "simulation").write_text(f"Remv {removed} [{version}]\n")
                admission = "" if case == "missing-admission" else str(root)
                selected = "" if case == "unselected-archive" else str(archive)
                result = run_bash(f'''
source "$1"
BOARD_PLATFORM_ADMISSION_DIR="{admission}"
FORCE_INSTALL=1
SKIP_PLATFORM_CHECK=ON
dpkg-query() {{ printf '%s' '{version}'; }}
verify_simulated_package_removals "{root / 'simulation'}" "{selected}"
''')
                self.assertEqual(result.returncode == 0, accepted, result.stdout + result.stderr)
                if accepted:
                    self.assertIn("neat-ev74-firmware=" + version, result.stdout)
                else:
                    self.assertIn("Refusing to install", result.stderr)


class RunnerRetirementTest(unittest.TestCase):
    def test_retirement_requires_relationships_from_file_owner_not_execution_provider(self):
        for removed in ("neat-runtime", "neat-gst-plugins"):
            for missing_owner in ("runner", "scheduler"):
                if removed == "neat-gst-plugins" and missing_owner == "runner":
                    continue  # Scheduler alone owns the plugin replacement.
                for missing_field in ("Breaks", "Replaces"):
                    with self.subTest(removed=removed, owner=missing_owner, field=missing_field):
                        result = run_bash(rf'''
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf '%s\n' '{{"platform-version":"3.0.0"}}' > "$tmp/manifest.json"
resolve_package_manifest_path() {{ printf '%s' "$tmp/manifest.json"; }}
touch "$tmp/runner.deb" "$tmp/scheduler.deb"
printf '%s\n' 'Remv {removed} [0.6.0+preview.arbitrary]' > "$tmp/simulation"
dpkg-query() {{ printf '%s' '0.6.0+preview.arbitrary'; }}
dpkg-deb() {{
  [[ "$2:$3" != *'/{missing_owner}.deb:{missing_field}' ]] || return 1
  case "$3" in
    Package) if [[ "$2" == *scheduler.deb ]]; then printf simaai-scheduler; else printf simaai-runner; fi ;;
    Breaks|Replaces) printf '%s' '{removed} (<< 0.6.1~)' ;;
    *) return 1 ;;
  esac
}}
verify_simulated_package_removals "$tmp/simulation" "$tmp/runner.deb" "$tmp/scheduler.deb"
''')
                        self.assertEqual(result.returncode == 0, missing_owner == "runner",
                                         result.stdout + result.stderr)

    def test_only_bounded_native_retirement_is_allowed(self):
        cases = [
            ("neat-runtime", "0.5.0+3.0.0-prep.abc", "3.0.0", "simaai-runtime", "<< 0.6.1~", False),
            ("neat-gst-plugins", "0.4.0+local.abc", "3.0.0", "simaai-runtime", "<< 0.6.1~", False),
            ("neat-runtime", "0.6.0", "3.0.0", "simaai-runtime", "<< 0.6.1~", False),
            ("neat-runtime", "0.5.0", "2.1.3", "simaai-runtime", "<< 0.6.1~", False),
            ("neat-runtime", "0.5.0", "3.0.0", "other-runtime", "<< 0.6.1~", False),
            ("neat-runtime", "0.5.0", "3.0.0", "simaai-runtime", "<< 9.0", False),
            ("simaai-palette-modalix", "0.5.0", "3.0.0", "simaai-runtime", "<< 0.6.1~", False),
        ]
        cases.extend([
            ("neat-gst-plugins", "0.4.0+local.abc", "3.0.0", "simaai-scheduler", "<< 0.6.1~", True),
            ("neat-gst-plugins", "0.5.0", "3.0.0", "simaai-scheduler", "<< 0.6.1~", True),
            ("neat-gst-plugins", "0.6.0", "3.0.0", "simaai-scheduler", "<< 0.6.1~", True),
            ("neat-gst-plugins", "0.6.0+develop.ce6b01ccdc94", "3.0.0", "simaai-scheduler", "<< 0.6.1~", True),
            ("neat-gst-plugins", "0.6.0+preview.arbitrary", "3.0.0", "simaai-scheduler", "<< 0.6.1~", True),
            ("neat-gst-plugins", "0.6.1~rc1", "3.0.0", "simaai-scheduler", "<< 0.6.1~", False),
            ("neat-gst-plugins", "0.6.1", "3.0.0", "simaai-scheduler", "<< 0.6.1~", False),
            ("neat-gst-plugins", "0.6.0", "2.1.3", "simaai-scheduler", "<< 0.6.1~", False),
            ("neat-gst-plugins", "0.4.0", "3.0.0", "simaai-scheduler", "<< 0.6~", True),
            ("neat-gst-plugins", "0.5.0+preview.arbitrary", "3.0.0", "simaai-scheduler", "<< 0.6~", True),
            ("neat-gst-plugins", "0.6.0", "3.0.0", "simaai-scheduler", "<< 0.6~", False),
            ("neat-runtime", "0.5.0", "3.0.0", "simaai-runner", "<< 0.6.1~", False),
            ("neat-runtime", "0.6.0", "3.0.0", "simaai-scheduler", "<< 0.6.1~", False),
            ("neat-gst-plugins", "0.5.0", "3.0.0", "simaai-scheduler", "<< 9.0", False),
        ])
        for removed, version, platform, package, bound, accepted in cases:
            with self.subTest(removed=removed, version=version, platform=platform, package=package, bound=bound):
                result = run_bash(rf'''
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf '%s\n' '{{"platform-version":"{platform}"}}' > "$tmp/manifest.json"
resolve_package_manifest_path() {{ printf '%s' "$tmp/manifest.json"; }}
touch "$tmp/runner.deb"
printf '%s\n' 'Remv {removed} [{version}]' > "$tmp/simulation"
dpkg-query() {{ printf '%s' '{version}'; }}
dpkg-deb() {{
  case "$3" in
    Package) printf '%s' '{package}' ;;
    Breaks|Replaces) printf '%s' '{removed} ({bound})' ;;
    *) return 1 ;;
  esac
}}
verify_simulated_package_removals "$tmp/simulation" "$tmp/runner.deb"
''')
                self.assertEqual(result.returncode == 0, accepted, result.stdout + result.stderr)

    def test_split_support_retirement_requires_both_real_providers(self):
        for version, include_scheduler, bound in ((version, selected, bound)
                for version in ("0.4.0", "0.5.0", "0.6.0", "0.6.0+develop.ce6b01ccdc94",
                                "0.6.0+preview.arbitrary", "0.6.1~rc1", "0.6.1")
                for selected in (False, True)
                for bound in ("0.6~", "0.6.1~")):
            extra = '\"$tmp/scheduler.deb\"' if include_scheduler else ''
            result = run_bash(r'''
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf '%s\n' '{"platform-version":"3.0.0"}' > "$tmp/manifest.json"
resolve_package_manifest_path() { printf '%s' "$tmp/manifest.json"; }
touch "$tmp/runner.deb" "$tmp/scheduler.deb"
printf '%s\n' 'Remv neat-runtime [0.5.0]' > "$tmp/simulation"
dpkg-query() { printf '%s' '0.5.0'; }
dpkg-deb() {
  case "$3" in
    Package) if [[ "$2" == *scheduler.deb ]]; then printf simaai-scheduler; else printf simaai-runner; fi ;;
    Breaks|Replaces)
      # Runner supplies execution, not the old Scheduler-owned files. Its
      # presence must suffice even without a legacy retirement declaration.
      [[ "$2" == *scheduler.deb ]] || return 1
      printf '%s' 'neat-runtime (<< 0.6.1~)' ;;
    *) return 1 ;;
  esac
}
verify_simulated_package_removals "$tmp/simulation" "$tmp/runner.deb" '''.replace("0.5.0", version).replace("0.6.1~", bound) + extra)
            self.assertEqual(result.returncode == 0,
                             include_scheduler and not version.startswith("0.6.1")
                             and (bound == "0.6.1~" or version.startswith(("0.4", "0.5"))),
                             (version, include_scheduler, bound, result.stdout + result.stderr))


class InstalledCoreLibraryLayoutTest(unittest.TestCase):
    def test_package_manifest_selects_library_directory(self) -> None:
        for layout in ("lib", "lib64", "lib/aarch64-linux-gnu"):
            with self.subTest(layout=layout):
                result = run_bash(r'''
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
fixture_dir="${tmp}/usr/''' + layout + r'''"
mkdir -p "${fixture_dir}"
touch "${fixture_dir}/libsima_neat.so.3.0.0"
# Exercise the real discovery and repair functions, not a libdir override.
dpkg-query() {
  [[ "$1" == '-L' && "$2" == 'sima-neat' ]] || return 1
  printf '%s\n' "${fixture_dir}/libsima_neat.so.3.0.0" \
    "${fixture_dir}/libsima_neat.so.4" "${fixture_dir}/libsima_neat.so"
}
read_sima_neat_elf_soname() { printf '%s\n' 'libsima_neat.so.4'; }
run_sudo() { "$@"; }
[[ "$(sima_neat_global_lib_dir)" == "${fixture_dir}" ]]
repair_global_sima_neat_lib_links
verify_global_sima_neat_lib_links
[[ "$(readlink "${fixture_dir}/libsima_neat.so.4")" == 'libsima_neat.so.3.0.0' ]]
[[ "$(readlink "${fixture_dir}/libsima_neat.so")" == 'libsima_neat.so.4' ]]
printf 'LAYOUT_OK\n'
''')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("LAYOUT_OK", result.stdout)

    def test_multiple_versioned_paths_are_rejected_before_repair(self) -> None:
        result = run_bash(r'''
source "$1"
dpkg-query() {
  [[ "$1" == '-L' && "$2" == 'sima-neat' ]] || return 1
  printf '%s\n' /usr/lib/libsima_neat.so.3.0.0 \
    /usr/lib/aarch64-linux-gnu/libsima_neat.so.3.0.0
}
run_sudo() { printf 'UNEXPECTED_MUTATION\n'; return 1; }
repair_global_sima_neat_lib_links
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly one packaged versioned", result.stderr)
        self.assertNotIn("UNEXPECTED_MUTATION", result.stdout)

    def test_soname_must_belong_to_selected_directory(self) -> None:
        result = run_bash(r'''
source "$1"
dpkg-query() {
  [[ "$1" == '-L' && "$2" == 'sima-neat' ]] || return 1
  printf '%s\n' /usr/lib/aarch64-linux-gnu/libsima_neat.so.3.0.0 \
    /usr/lib/libsima_neat.so.4
}
run_sudo() { printf 'UNEXPECTED_MUTATION\n'; return 1; }
repair_global_sima_neat_lib_links
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly one packaged libsima_neat SONAME", result.stderr)
        self.assertNotIn("UNEXPECTED_MUTATION", result.stdout)


class BoardExactReinstallTest(unittest.TestCase):
    SETUP = r'''
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
refresh_apt_metadata_for_board_install() { :; }
stop_board_runtime_before_install() { :; }
apt_package_database_is_healthy() { return 0; }
collect_board_heal_specs() { :; }
complete_board_install_after_packages() { :; }
board_runtime_is_legacy() { return 0; }
'''

    def test_exact_architecture_version_specs_are_deduplicated(self) -> None:
        result = run_bash(self.SETUP + r'''
DEBS=(./runtime.deb ./headers.deb ./runtime.deb)
dpkg-deb() {
  case "$2:$3" in
    ./runtime.deb:Package) echo example-runtime ;;
    ./headers.deb:Package) echo example-headers ;;
    *:Version) echo 1.2.3 ;;
    ./runtime.deb:Architecture) echo arm64 ;;
    ./headers.deb:Architecture) echo all ;;
    *) return 1 ;;
  esac
}
run_sudo() { printf '%s\n' "$*" >> "${tmp}/commands"; }
install_debs_on_board
cat "${tmp}/commands"
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = [line for line in result.stdout.splitlines() if line.startswith("apt-get install")]
        self.assertEqual(len(commands), 2)
        for command in commands:
            for spec in ("./runtime.deb", "./headers.deb", "example-runtime:arm64=1.2.3", "example-headers=1.2.3"):
                self.assertEqual(command.split().count(spec), 1)
        self.assertIn("--simulate", commands[0])
        self.assertNotIn("--simulate", commands[1])
        for command in commands:
            self.assertNotIn("--force-overwrite", command)

    OWNERSHIP_SETUP = SETUP + r'''
NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=OFF
NEAT_INSTALLER_ALLOW_DPKG_FALLBACK=OFF
DEBS=(./runtime.deb)
dpkg-deb() {
  case "$3" in
    Package) echo example-runtime ;;
    Version) echo 1.2.3 ;;
    Architecture) echo arm64 ;;
    *) return 1 ;;
  esac
}
printf 'original-owner-bytes\n' > "${tmp}/owned-file"
: > "${tmp}/commands"
trap 'cat "${tmp}/commands"; printf "OWNER_CONTENT="; cat "${tmp}/owned-file"; rm -rf "${tmp}"' EXIT
remove_installed_local_deb_packages() { printf 'REMOVE_LOCAL_PACKAGES\n' >> "${tmp}/commands"; }
complete_board_install_after_packages() { printf 'COMPLETE_INSTALL\n' >> "${tmp}/commands"; }
'''

    def test_undeclared_file_collision_fails_without_default_removal(self) -> None:
        # The removal simulation is clean: the colliding owner is not removed.
        # The shell double models dpkg's later file-ownership rejection, and
        # deliberately accepts the unsafe overwrite option as a mutation control.
        result = run_bash(self.OWNERSHIP_SETUP + r'''
run_sudo() {
  printf '%s\n' "$*" >> "${tmp}/commands"
  if [[ " $* " == *" --simulate "* ]]; then
    return 0
  fi
  if [[ "$1 $2" == 'apt-get install' ]]; then
    if [[ "$*" == *--force-overwrite* ]]; then
      printf 'overwritten-by-new-owner\n' > "${tmp}/owned-file"
      return 0
    fi
    echo 'dpkg: trying to overwrite a file owned by unrelated-package' >&2
    return 100
  fi
  [[ "$1 $2" == 'apt-get check' ]] && return 0
  echo 'UNEXPECTED_PACKAGE_COMMAND' >&2
  return 91
}
install_debs_on_board
''')
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("trying to overwrite", result.stderr)
        self.assertIn("refusing to remove the installed runtime", result.stderr)
        self.assertIn("OWNER_CONTENT=original-owner-bytes", result.stdout)
        self.assertNotIn("--force-overwrite", result.stdout)
        self.assertNotIn("REMOVE_LOCAL_PACKAGES", result.stdout)
        self.assertNotIn("COMPLETE_INSTALL", result.stdout)
        self.assertNotIn("dpkg -i", result.stdout)
        self.assertNotIn("UNEXPECTED_PACKAGE_COMMAND", result.stderr)

    def test_explicit_direct_fallback_still_rejects_file_collision(self) -> None:
        # Explicit maintenance permits only the existing bundled-package removal
        # path. It must not disable ownership checks in the direct dpkg fallback.
        result = run_bash(self.OWNERSHIP_SETUP + r'''
NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=ON
NEAT_INSTALLER_ALLOW_DPKG_FALLBACK=ON
run_sudo() {
  printf '%s\n' "$*" >> "${tmp}/commands"
  if [[ " $* " == *" --simulate "* ]]; then
    return 0
  fi
  if [[ "$1 $2" == 'apt-get install' ]]; then
    echo 'APT transaction failure' >&2
    return 100
  fi
  if [[ "$1 $2" == 'dpkg -i' ]]; then
    if [[ "$*" == *--force-overwrite* ]]; then
      printf 'overwritten-by-new-owner\n' > "${tmp}/owned-file"
      return 0
    fi
    echo 'dpkg: trying to overwrite a file owned by unrelated-package' >&2
    return 73
  fi
  [[ "$1 $2" == 'apt-get check' ]] && return 0
  echo 'UNEXPECTED_PACKAGE_COMMAND' >&2
  return 91
}
install_debs_on_board
''')
        self.assertEqual(result.returncode, 73, result.stdout + result.stderr)
        self.assertIn("trying to overwrite", result.stderr)
        self.assertIn("OWNER_CONTENT=original-owner-bytes", result.stdout)
        self.assertIn("dpkg -i ./runtime.deb", result.stdout)
        self.assertEqual(result.stdout.count("REMOVE_LOCAL_PACKAGES"), 1)
        self.assertNotIn("--force-overwrite", result.stdout)
        self.assertNotIn("COMPLETE_INSTALL", result.stdout)
        self.assertNotIn("UNEXPECTED_PACKAGE_COMMAND", result.stderr)

    def test_conflicting_local_versions_fail_before_apt(self) -> None:
        result = run_bash(self.SETUP + r'''
DEBS=(./one.deb ./two.deb)
dpkg-deb() {
  case "$3" in
    Package) echo example-runtime ;;
    Architecture) echo arm64 ;;
    Version) [[ "$2" == ./one.deb ]] && echo 1 || echo 2 ;;
    *) return 1 ;;
  esac
}
run_sudo() { printf 'UNEXPECTED_APT\n'; }
install_debs_on_board
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Conflicting local DEB versions", result.stderr)
        self.assertNotIn("UNEXPECTED_APT", result.stdout)

    def test_real_apt_simulation_reinstalls_same_version_local_deb(self) -> None:
        # A private dpkg status and empty source lists keep this behavioral
        # probe read-only. No package is installed, and no target code runs.
        result = run_bash(self.SETUP + r'''
command -v apt-get >/dev/null
command -v dpkg-deb >/dev/null
mkdir -p "${tmp}/package/DEBIAN" "${tmp}/lists/partial" "${tmp}/cache/archives/partial" "${tmp}/sources"
cat > "${tmp}/package/DEBIAN/control" <<'CONTROL'
Package: neat-reinstall-probe
Version: 1.0
Architecture: all
Maintainer: Test <test@example.invalid>
Description: local reinstall simulation fixture
CONTROL
{ cat "${tmp}/package/DEBIAN/control"; printf 'Status: install ok installed\n'; } > "${tmp}/status"
dpkg-deb --build --root-owner-group "${tmp}/package" "${tmp}/probe.deb" >/dev/null
DEBS=("${tmp}/probe.deb")
run_sudo() {
  if [[ " $* " == *" --simulate "* ]]; then
    shift
    command apt-get -o "Dir::State::status=${tmp}/status" \
      -o "Dir::State::lists=${tmp}/lists" -o "Dir::Cache=${tmp}/cache" \
      -o Dir::Etc::sourcelist=/dev/null -o "Dir::Etc::sourceparts=${tmp}/sources" \
      -o APT::Get::Download=false -o APT::Get::Remove=false "$@" > "${tmp}/simulation"
    cat "${tmp}/simulation"
  fi
  # Deliberately do not execute the non-simulated transaction or apt check.
}
install_debs_on_board
cat "${tmp}/simulation"
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Inst neat-reinstall-probe [1.0] (1.0 local-deb", result.stdout)
        self.assertIn("1 reinstalled", result.stdout)


class DirectPlatformPreservationTest(unittest.TestCase):
    """Palette 3 identity is its native package, not the removed OTA command."""

    def probe(self, script: str) -> subprocess.CompletedProcess[str]:
        return run_bash(r'''
source "$1"
version=1
architecture=arm64
status="install ok installed"
board_runtime_is_legacy() { return 1; }
deb_package_is_installed() { [[ "$status" == "install ok installed" ]]; }
dpkg-query() {
  [[ "$1" == -W ]] || { echo UNEXPECTED_FILE_DISCOVERY >&2; return 91; }
  printf 'simaai-palette-modalix\t%s\t%s\t%s\n' "$version" "$architecture" "$status"
}
repair_global_sima_neat_lib_links() { :; }
verify_global_sima_neat_lib_links() { :; }
activate_board_runtime_after_install() { :; }
restart_board_codec_services() { :; }
verify_board_codec_services() { :; }
''' + script)

    def test_installed_palette_needs_no_ota_executable(self) -> None:
        result = self.probe(r'''
before="$(capture_direct_platform_identity)"
complete_board_install_after_packages "$before"
printf '%s\n' "$before"
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("simaai-palette-modalix\t1\tarm64\tinstall ok installed", result.stdout)
        self.assertNotIn("UNEXPECTED_FILE_DISCOVERY", result.stderr)

    def test_missing_palette_is_rejected(self) -> None:
        result = self.probe('status="deinstall ok config-files"\nverify_canonical_palette_installation\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not installed", result.stderr)

    def test_post_install_package_identity_changes_are_rejected(self) -> None:
        for change in ("version=2", "architecture=amd64"):
            with self.subTest(change=change):
                result = self.probe('before="$(capture_direct_platform_identity)"\n' + change +
                                    '\ncomplete_board_install_after_packages "$before"\n')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Native platform identity changed", result.stderr)

    def test_lost_package_ownership_is_rejected(self) -> None:
        result = self.probe(r'''
before="$(capture_direct_platform_identity)"
status="deinstall ok config-files"
complete_board_install_after_packages "$before"
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not installed", result.stderr)

    def test_package_query_failure_is_not_empty_success(self) -> None:
        result = self.probe('dpkg-query() { return 17; }\ncapture_direct_platform_identity\n')
        self.assertEqual(result.returncode, 17)


class PlatformAdmissionBoundaryTest(unittest.TestCase):
    def test_dpkg_completion_signal_uses_machine_phases_not_dependency_health(self) -> None:
        for states, expected in (("install ok installed\nunknown ok not-installed\ndeinstall ok config-files", 0),
                                 ("hold ok installed", 0), ("install ok installed\ninstall ok unpacked", 1),
                                 ("install reinstreq half-installed", 1), ("install ok triggers-pending", 1),
                                 ("install reinstreq installed", 1), ("deinstall ok installed", 1),
                                 ("install ok unknown-phase", 1)):
            with self.subTest(states=states):
                result = run_bash('source "$1"\ndpkg-query() { printf "%s\\n" "' + states +
                                  '"; }\ndpkg_package_database_is_configured\n')
                self.assertEqual(result.returncode, expected, result.stderr)
        failed = run_bash('source "$1"\ndpkg-query() { echo "query failed" >&2; return 17; }\n'
                          'dpkg_package_database_is_configured\n')
        self.assertEqual(failed.returncode, 2)
        self.assertIn("query failed", failed.stderr)

    def test_only_interim_completion_allows_normal_pending_triggers(self) -> None:
        for state in ("install ok triggers-pending", "install ok triggers-awaited",
                      "hold ok triggers-pending", "deinstall ok triggers-awaited",
                      "install reinstreq triggers-pending", "install ok unpacked",
                      "install ok half-configured", "install reinstreq installed"):
            for mode in ("", "before-reinstall"):
                with self.subTest(state=state, mode=mode):
                    result = run_bash('source "$1"\ndpkg-query() { '
                        '[[ "$*" == \'-W -f=${Status}\\n\' ]] || return 17; '
                        'printf "%s\\n" "' + state + '"; }\n'
                        'dpkg_package_database_is_configured ' + mode + '\n')
                    allowed = mode == "before-reinstall" and state in (
                        "install ok triggers-pending", "install ok triggers-awaited")
                    self.assertEqual(result.returncode, 0 if allowed else 1, result.stderr)
        failed = run_bash('source "$1"\ndpkg-query() { return 17; }\n'
                          'dpkg_package_database_is_configured before-reinstall\n')
        self.assertEqual(failed.returncode, 2)

    @staticmethod
    def trigger_resume_probe(interim: str, final: str = "install ok installed", *,
                             handler_fails: bool = False) -> subprocess.CompletedProcess[str]:
        # Real shell flow/classifier; external APT and producer decisions are
        # controlled doubles. Scheduler tests own real trigger/plan admission.
        return run_bash(r'''
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
cd "${tmp}"
DEBS=(./candidate.deb)
APPLIES=0
prepare_board_platform_admission() { echo PLATFORM; }
verify_board_platform_admission() { echo PLATFORM; }
verify_board_platform_plan() { :; }
verify_simulated_package_removals() { :; }
verify_scheduler_transaction() { shift; echo "ADMISSION $*"; }
board_runtime_is_legacy() { return 0; }
refresh_apt_metadata_for_board_install() { :; }
stop_board_runtime_before_install() { :; }
board_has_scheduler_cohort() { return 0; }
apt_package_database_is_healthy() { return 0; }
collect_exact_deb_install_specs() { local -n specs="$1"; specs=(./candidate.deb candidate:arm64=1); }
collect_board_heal_specs() { :; }
complete_board_install_after_packages() { echo COMPLETE; }
remove_installed_local_deb_packages() { echo UNEXPECTED_REMOVAL; }
NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=ON
NEAT_INSTALLER_ALLOW_DPKG_FALLBACK=ON
dpkg-query() {
  [[ "$*" == '-W -f=${Status}\n' || "$*" == '-W -f=${db:Status-Status}\n' ]] || return 17
  local phase_only=0
  [[ "$*" != '-W -f=${db:Status-Status}\n' ]] || phase_only=1
  local state
  case "${APPLIES}" in
    0) state='install ok triggers-pending' ;;
    1) [[ "${INTERIM}" != QUERY_FAILED ]] || return 17; state="${INTERIM}" ;;
    *) [[ "${FINAL}" != QUERY_FAILED ]] || return 17; state="${FINAL}" ;;
  esac
  [[ "${phase_only}" != 1 ]] || state="${state##* }"
  printf '%s\n' "${state}"
}
python3() {
  [[ "$1" == -B && "$2" == "${SCHEDULER_PLATFORM_ADMISSION_PATH}" && "$3" == retirements ]]
}
run_sudo() {
  printf '%s\n' "$*" >> commands
  if [[ " $* " == *' --simulate '* ]]; then
    # The first resume is a no-op with queued triggers; only the mandatory
    # strict reinstall must plan all selected package Inst/Conf actions.
    if [[ " $* " == *' --reinstall '* ]]; then
      echo 'Inst candidate (1 local-deb [arm64])'
      echo 'Conf candidate (1 local-deb [arm64])'
    fi
    return 0
  fi
  if [[ "$1 $2" == 'apt-get install' ]]; then
    APPLIES=$((APPLIES + 1))
    echo "APPLY ${APPLIES}"
    if [[ "${APPLIES}" == 2 && "${HANDLER_FAILS}" == 1 ]]; then
      echo 'dpkg trigger handler failed' >&2
      return 100
    fi
    return 0
  fi
  [[ "$*" == 'apt-get check' ]]
}
''' + f"INTERIM='{interim}'; FINAL='{final}'; HANDLER_FAILS={int(handler_fails)}\n" + r'''
if install_debs_on_board; then result=0; else result=$?; fi
cat commands
exit "${result}"
''')

    def test_trigger_only_resume_still_requires_strict_reinstall(self) -> None:
        for state in ("install ok triggers-pending", "install ok triggers-awaited"):
            with self.subTest(state=state):
                result = self.trigger_resume_probe(state)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("APPLY 1\n", result.stdout)
                self.assertIn("APPLY 2\n", result.stdout)
                self.assertIn("ADMISSION --repair-incomplete --resume-incomplete\n", result.stdout)
                self.assertIn("ADMISSION --repair-incomplete\n", result.stdout)
                self.assertIn("COMPLETE\n", result.stdout)
                phases = [line for line in result.stdout.splitlines()
                          if line == "PLATFORM" or line.startswith(("ADMISSION ", "APPLY "))]
                self.assertEqual(phases, ["PLATFORM", "ADMISSION --repair-incomplete --resume-incomplete",
                                          "APPLY 1", "PLATFORM", "ADMISSION --repair-incomplete", "APPLY 2"])
                commands = [s for s in result.stdout.splitlines() if s.startswith("apt-get install")]
                self.assertEqual(len(commands), 4)
                for index, command in enumerate(commands):
                    self.assertIn("--no-remove", command)
                    self.assertEqual("--reinstall" in command, index >= 2)
                    self.assertIn("./candidate.deb candidate:arm64=1", command)

    def test_nontrigger_interim_states_cannot_reach_strict_reinstall(self) -> None:
        for state in ("install ok unpacked", "install ok half-configured", "hold ok triggers-pending",
                      "install reinstreq triggers-pending", "install reinstreq installed", "QUERY_FAILED"):
            with self.subTest(state=state):
                result = self.trigger_resume_probe(state)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("APPLY 1\n", result.stdout)
                for forbidden in ("APPLY 2", "COMPLETE", "UNEXPECTED_REMOVAL", "dpkg -i", "--reinstall"):
                    self.assertNotIn(forbidden, result.stdout)

    def test_final_pending_state_or_trigger_handler_failure_prevents_completion(self) -> None:
        for state, fails in (("install ok triggers-pending", False), ("install ok triggers-awaited", False),
                             ("install ok half-configured", False), ("install reinstreq installed", False),
                             ("QUERY_FAILED", False), ("install ok installed", True)):
            with self.subTest(state=state, handler_fails=fails):
                result = self.trigger_resume_probe("install ok triggers-pending", state, handler_fails=fails)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("APPLY 2\n", result.stdout)
                for forbidden in ("COMPLETE", "UNEXPECTED_REMOVAL", "dpkg -i"):
                    self.assertNotIn(forbidden, result.stdout)
                if fails:
                    self.assertIn("dpkg trigger handler failed", result.stderr)
                    self.assertIn("Final recovery reinstall failed", result.stderr)

    def test_unhealthy_apt_diagnostic_is_preserved_before_temporary_log_cleanup(self) -> None:
        for status in (0, 100):
            with self.subTest(status=status):
                result = run_bash(r'''
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
mktemp() { command mktemp "${tmp}/apt-check-XXXXXX"; }
run_sudo() { echo 'Exact APT dependency diagnostic' >&2; return ''' + str(status) + r'''; }
dpkg-query() { echo 'UNEXPECTED_DPKG_DIAGNOSTIC' >&2; return 17; }
if apt_package_database_is_healthy; then echo HEALTHY; else echo UNHEALTHY; fi
[[ -z "$(find "${tmp}" -type f -print)" ]]
''')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "HEALTHY" if status == 0 else "UNHEALTHY")
                self.assertEqual("Exact APT dependency diagnostic" in result.stderr, status != 0)
                self.assertEqual("UNEXPECTED_DPKG_DIAGNOSTIC" in result.stderr, status != 0)
                self.assertEqual("Unable to read incomplete dpkg state" in result.stderr, status != 0)

    def test_unhealthy_apt_reports_exact_incomplete_states_and_trigger_fields(self) -> None:
        rows = [
            "configured\tinstall ok installed\tii \t1\tarm64\t\t",
            "held-configured\thold ok installed\thi \t1\tarm64\t\t",
            "absent\tunknown ok not-installed\tun \t\t\t\t",
            "removed\tdeinstall ok config-files\trc \t1\tarm64\t\t",
            "trigger-owner:arm64\tinstall ok triggers-pending\tit \t2.3-1\tarm64\tldconfig other-trigger\t",
            "awaiting-client\tinstall ok triggers-awaited\tiW \t4+preview\tall\t\ttrigger-owner",
            "unpacked-client\tinstall ok unpacked\tiU \t4\tarm64\t\t",
            "broken-client\tinstall reinstreq half-installed\tiHR\t5\tarm64\t\t",
            "half-configured\tinstall ok half-configured\tiF \t6\tarm64\t\t",
            "held-pending\thold ok triggers-pending\tht \t7\tarm64\tldconfig\t",
            "abnormal-selection\tdeinstall ok installed\tri \t8\tarm64\t\t",
            "abnormal-error\tinstall reinstreq installed\tiiR\t9\tarm64\t\t",
        ]
        result = run_bash(r'''
source "$1"
run_sudo() { [[ "$*" == 'apt-get check' ]] || exit 99; echo 'Original APT failure' >&2; return 100; }
dpkg-query() {
  [[ "$#" == 2 && "$1" == -W ]] || exit 98
  [[ "$2" == '-f=${binary:Package}\t${Status}\t${db:Status-Abbrev}\t${Version}\t${Architecture}\t${Triggers-Pending}\t${Triggers-Awaited}\n' ]] || exit 97
  cat <<'STATES'
''' + "\n".join(rows) + r'''
STATES
}
apt_package_database_is_healthy
echo UNEXPECTED_SUCCESS
''')
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertNotIn("UNEXPECTED_SUCCESS", result.stdout)
        self.assertIn("Original APT failure", result.stderr)
        for row in rows[:4]:
            self.assertNotIn(row, result.stderr)
        for row in rows[4:]:
            self.assertIn(row + "\n", result.stderr)
        self.assertIn("Incomplete dpkg records: 8; omitted: 0", result.stderr)
        self.assertLess(result.stderr.index("Original APT failure"),
                        result.stderr.index("trigger-owner:arm64"))

    def test_unhealthy_apt_dpkg_diagnostic_is_bounded(self) -> None:
        result = run_bash(r'''
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
cd "${tmp}"
run_sudo() { echo 'Original APT failure' >&2; return 100; }
dpkg-query() {
  for ((i=1; i<=103; ++i)); do
    printf 'pending-%03d\tinstall ok unpacked\tiU \t1\tarm64\t\t\n' "$i"
  done
}
if apt_package_database_is_healthy; then status=0; else status=$?; fi
[[ -z "$(find . -type f -print)" ]] || { echo UNEXPECTED_DIAGNOSTIC_WRITE >&2; exit 99; }
exit "${status}"
''')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(sum(line.startswith("pending-") for line in result.stderr.splitlines()), 100)
        self.assertIn("pending-100\t", result.stderr)
        self.assertNotIn("pending-101\t", result.stderr)
        self.assertIn("Incomplete dpkg records: 103; omitted: 3", result.stderr)

    def test_unhealthy_dependencies_need_not_have_incomplete_dpkg_records(self) -> None:
        result = run_bash(r'''
source "$1"
run_sudo() { echo 'Original APT dependency failure' >&2; return 100; }
dpkg-query() { printf 'configured\tinstall ok installed\tii \t1\tarm64\t\t\n'; }
apt_package_database_is_healthy
''')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Original APT dependency failure", result.stderr)
        self.assertIn("Incomplete dpkg records: 0; omitted: 0", result.stderr)
        self.assertNotIn("configured\t", result.stderr)

    def test_failed_dpkg_diagnostic_preserves_original_apt_failure_status(self) -> None:
        for failing_command in ("dpkg-query", "awk"):
            with self.subTest(failing_command=failing_command):
                result = run_bash(r'''
source "$1"
run_sudo() { echo 'Original APT failure' >&2; return 100; }
dpkg-query() { printf 'pending\tinstall ok unpacked\tiU \t1\tarm64\t\t\n'; }
''' + failing_command + r'''() { echo 'Read-only diagnostic failed' >&2; return 17; }
apt_package_database_is_healthy
echo UNEXPECTED_SUCCESS
''')
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("Original APT failure", result.stderr)
                self.assertIn("Read-only diagnostic failed", result.stderr)
                self.assertNotIn("UNEXPECTED_SUCCESS", result.stdout)

    def test_incomplete_state_diagnostic_does_not_admit_board_package_mutation(self) -> None:
        for query_fails in (False, True):
            with self.subTest(query_fails=query_fails):
                result = run_bash(r'''
source "$1"
DEBS=(./candidate.deb)
prepare_board_platform_admission() { :; }
verify_board_platform_admission() { :; }
board_runtime_is_legacy() { return 0; }
refresh_apt_metadata_for_board_install() { :; }
stop_board_runtime_before_install() { :; }
board_has_scheduler_cohort() { return 0; }
collect_exact_deb_install_specs() { local -n specs="$1"; specs=(./candidate.deb); }
collect_board_heal_specs() { :; }
complete_board_install_after_packages() { echo UNEXPECTED_LIFECYCLE; }
run_sudo() {
  if [[ "$*" == 'apt-get check' ]]; then
    echo 'Original APT failure' >&2
    return 100
  fi
  echo "UNEXPECTED_MUTATION $*"
  return 99
}
dpkg-query() {
  if [[ "$2" == '-f=${Status}\n' ]]; then
    echo 'install ok triggers-pending'
    return 0
  fi
''' + ("  echo 'Read-only diagnostic failed' >&2; return 17\n" if query_fails else
       "  printf 'foreign-pending\\tinstall ok triggers-pending\\tit \\t1\\tarm64\\tldconfig\\t\\n'\n") + r'''
}
python3() {
  [[ "$1" == -B && "$2" == "${SCHEDULER_PLATFORM_ADMISSION_PATH}" && "$3" == retirements ]] || return 98
  echo 'PRODUCER_REFUSED_FOREIGN_PENDING' >&2
  return 23
}
install_debs_on_board
echo UNEXPECTED_SUCCESS
''')
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("Original APT failure", result.stderr)
                self.assertIn("PRODUCER_REFUSED_FOREIGN_PENDING", result.stderr)
                self.assertNotIn("UNEXPECTED_", result.stdout)
                self.assertEqual("Read-only diagnostic failed" in result.stderr, query_fails)
                if not query_fails:
                    self.assertIn("foreign-pending\tinstall ok triggers-pending", result.stderr)

    def test_version_overrides_still_require_provider_admission_before_neat_apt(self) -> None:
        for override in ("FORCE_INSTALL=1", "NEAT_INSTALLER_SKIP_PLATFORM_CHECK=ON"):
            with self.subTest(override=override):
                result = run_bash('source "$1"\nENV_MODE=modalix-board\n' + override +
                                  '\nDEBS=(./candidate.deb)'
                                  '\nprepare_board_platform_admission() { echo ADMISSION; return 9; }'
                                  '\nrun_sudo() { echo UNEXPECTED_APT; }'
                                  '\nensure_platform_compatible\necho VERSION_CHECK_FINISHED'
                                  '\ninstall_debs_on_board\necho UNEXPECTED_SUCCESS\n')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ADMISSION", result.stdout)
                self.assertLess(result.stdout.index("VERSION_CHECK_FINISHED"), result.stdout.index("ADMISSION"))
                self.assertNotIn("UNEXPECTED_", result.stdout)

    def test_sdk_modes_do_not_use_host_as_board_provider_authority(self) -> None:
        for mode in ("elxr-sdk", "ros2-sdk"):
            with self.subTest(mode=mode):
                result = run_bash('source "$1"\nFORCE_INSTALL=1\nENV_MODE=' + mode +
                                  '\nprepare_board_platform_admission() { echo ADMISSION; return 9; }'
                                  '\nensure_platform_compatible\n')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("ADMISSION", result.stdout)


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
    unittest.main()
