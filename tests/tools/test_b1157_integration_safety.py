"""Host-only B1157 pairing and recovery regressions; never touch a DevKit."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "deps/manifest.json").read_text())


def bash(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=30)


def test_native_recovery_policy_and_direct_command_guard(tmp_path):
    binary = tmp_path / "recovery-policy"
    subprocess.run([
        "/usr/bin/g++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
        "-DSIMA_NEAT_INTERNAL=1", "-DSIMANEAT_DIRECT_DRIVER_PROFILE=1",
        "-DNEAT_TEST_WRAP_SYSTEM=1", "-Wl,--wrap=system",
        "-I", str(ROOT / "src"), "-I", str(ROOT / "include"),
        str(ROOT / "tests/unit_testing/unit_runtime_recovery_policy_test.cpp"),
        str(ROOT / "src/pipeline/gst/DispatcherRecovery.cpp"), "-o", str(binary),
    ], check=True, timeout=90)
    kind = subprocess.check_output(["file", str(binary)], text=True)
    assert "x86-64" in kind and "aarch64" not in kind, kind
    subprocess.run([str(binary)], check=True, timeout=10)


@pytest.mark.parametrize("metadata,allowed", [
    ("MACHINE=modalix\nDISTRO_VERSION=2.1.2\n", True),
    ("MACHINE = modalix\nDISTRO_VERSION = 2.1.3~pre4744\n", True),
    ("MACHINE=modalix\nDISTRO_VERSION=2.1.3+local\n", True),
    ("MACHINE=modalix\nDISTRO_VERSION=3.0.0\n", False),
    ("MACHINE=modalix\nDISTRO_VERSION=2.1.3garbage\n", False),
    ("MACHINE=modalix\nDISTRO_VERSION=2.1.3\nDISTRO_VERSION=3.0.0\n", False),
    ("MACHINE=davinci\nDISTRO_VERSION=2.1.3\n", False),
    ("", False),
])
def test_shell_profile_classification(tmp_path, metadata, allowed):
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc/buildinfo").write_text(metadata)
    result = bash(f"""
export NEAT_RECOVERY_FUNCTIONS_ONLY=ON
source {shlex.quote(str(ROOT / 'scripts/fix_devkit_runtime.sh'))}
legacy_runtime_recovery_allowed_at {shlex.quote(str(tmp_path))}
""")
    assert (result.returncode == 0) == allowed, result.stderr


def test_manifest_selects_internals_and_llima_artifacts():
    assert MANIFEST["platform-version"] == "3.0.0"
    for dependency in ("internals", "llima"):
        selection = MANIFEST[dependency]
        assert selection["branch"] == "codex/b1157-mlart-dmabuf"
        assert len(selection["spec"]) == 12
        assert all(character in "0123456789abcdef" for character in selection["spec"])
    assert "runtime-profile" not in MANIFEST
    assert "kernel-commit" not in MANIFEST
    assert "expected-internals-sysroot" not in MANIFEST


def test_direct_installer_requires_attestation_before_any_action():
    result = bash(f"""
source {shlex.quote(str(ROOT / 'tools/install_neat_framework.sh'))}
board_runtime_is_legacy() {{ return 1; }}
unset NEAT_INSTALLER_B1157_MAINTENANCE
run_sudo() {{ echo MUTATED; }}
install_python_environment() {{ echo MUTATED; }}
ENV_MODE=modalix-board
install_for_environment
""")
    assert result.returncode != 0
    assert "MUTATED" not in result.stdout
    assert "NEAT_INSTALLER_B1157_MAINTENANCE=confirmed" in result.stderr


@pytest.mark.parametrize("action", [
    "stop_board_runtime_before_install", "activate_board_runtime_after_install",
    "verify_board_runtime_services", "restart_board_codec_services", "verify_board_codec_services",
])
def test_direct_installer_never_calls_legacy_lifecycle(action):
    result = bash(f"""
source {shlex.quote(str(ROOT / 'tools/install_neat_framework.sh'))}
board_runtime_is_legacy() {{ return 1; }}
NEAT_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD=ON
run_sudo() {{ echo MUTATED; }}
systemctl() {{ echo MUTATED; }}
{action}
""")
    assert result.returncode == 0, result.stderr
    assert "MUTATED" not in result.stdout


def test_direct_postinstall_preserves_checks_without_dispatcher_or_activation():
    result = bash(f"""
source {shlex.quote(str(ROOT / 'tools/install_neat_framework.sh'))}
board_runtime_is_legacy() {{ return 1; }}
migrate_stale_global_dispatcher_libs() {{ echo LEGACY; }}
verify_private_dispatcher_runtime() {{ echo LEGACY; }}
repair_global_sima_neat_lib_links() {{ echo ELF_LINK_CHECK; }}
verify_global_sima_neat_lib_links() {{ echo ELF_VERIFY; }}
verify_canonical_palette_and_ota_installation() {{ echo PLATFORM_VERIFY; }}
run_sudo() {{ echo LEGACY; }}
systemctl() {{ echo LEGACY; }}
complete_board_install_after_packages
""")
    assert result.returncode == 0, result.stderr
    assert "LEGACY" not in result.stdout
    assert "ELF_LINK_CHECK" in result.stdout
    assert "ELF_VERIFY" in result.stdout
    assert "PLATFORM_VERIFY" in result.stdout


def test_modalix_3_llima_sysroot_dependencies_include_runtime_libraries():
    build = (ROOT / "build.sh").read_text()
    installer = (ROOT / "tools/install_neat_framework.sh").read_text()
    for package in (
        "libfmt10:arm64",
        "libspdlog1.15:arm64",
        "libcpp-httplib0.18:arm64",
        "libfftw3-double3:arm64",
        "libavcodec61:arm64",
        "libavformat61:arm64",
        "libavutil59:arm64",
        "libswresample5:arm64",
    ):
        assert package in build
        assert package in installer
    assert "libspdlog1.10:arm64" not in installer
    assert "--preserve-env=SDK_APT_CHANNEL" in build


@pytest.mark.parametrize("state,expected", [
    ("idle", True), ("active", False), ("enabled", False), ("owner", False), ("unknown", False),
])
def test_maintenance_preflight_is_read_only_and_rejects_busy_state(tmp_path, state, expected):
    commands = tmp_path / "bin"
    commands.mkdir()
    mocks = {
        "systemctl": '''#!/bin/bash
case "$1" in
is-active) [[ "$CHECK_STATE" == active ]] && echo active || echo inactive ;;
is-enabled) case "$CHECK_STATE" in enabled) echo enabled ;; unknown) echo bad-state ;; *) echo disabled ;; esac ;;
*) echo MUTATED; exit 10 ;;
esac
''',
        "pgrep": "#!/bin/bash\nexit 1\n",
        "fuser": '#!/bin/bash\n[[ "$CHECK_STATE" == owner ]] && exit 0\nexit 1\n',
    }
    for name, contents in mocks.items():
        path = commands / name
        path.write_text(contents)
        path.chmod(0o755)
    for name in ("mla", "cvu", "allegroIP", "allegroDecodeIP"):
        (tmp_path / name).touch()
    result = bash(rf"""
source {shlex.quote(str(ROOT / 'tools/install_neat_framework.sh'))}
board_runtime_is_legacy() {{ return 1; }}
export CHECK_STATE={state}
export PATH={shlex.quote(str(commands))}:/usr/bin:/bin
NEAT_INSTALLER_B1157_MAINTENANCE=confirmed
run_sudo() {{
  # Only the read-only preflight executes, against fixture paths and commands.
  if [[ "$1" == /usr/bin/neat-b1157-migration-check ]]; then [[ "$#" -eq 1 ]]; return; fi
  [[ "$1" == bash && "$2" == -c ]] || return 20
  local body="$3"
  body="${{body//\/dev\//{str(tmp_path)}/}}"
  bash -c "$body"
}}
check_b1157_install_maintenance
""")
    assert (result.returncode == 0) == expected, result.stderr
    assert "MUTATED" not in result.stdout


def test_legacy_installer_classification_rejects_duplicate_image_identity(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"platform-version":"2.1.3"}')
    buildinfo = tmp_path / "buildinfo"
    buildinfo.write_text("MACHINE=modalix\nDISTRO_VERSION=2.1.3\nDISTRO_VERSION=3.0.0\n")
    result = bash(f"""
source {shlex.quote(str(ROOT / 'tools/install_neat_framework.sh'))}
NEAT_PACKAGE_MANIFEST={shlex.quote(str(manifest))}
NEAT_BUILDINFO_FILE={shlex.quote(str(buildinfo))}
board_runtime_is_legacy
""")
    assert result.returncode != 0


def test_installer_uses_only_the_public_read_only_migration_entrypoint():
    installer = (ROOT / "tools/install_neat_framework.sh").read_text()
    assert "run_sudo /usr/bin/neat-b1157-migration-check || return 1" in installer
    assert "run_sudo /usr/bin/neat-b1157-migration-check check" not in installer
    assert "/usr/libexec/sima-neat/runtime-migration" not in installer


def test_maintenance_preflight_allows_platform_init_and_trace_services():
    installer = (ROOT / "tools/install_neat_framework.sh").read_text()
    function = installer.split("check_b1157_install_maintenance() {", 1)[1].split(
        "\n}\n", 1
    )[0]
    assert "simaai-appcomplex.service" not in function
    assert "rctd.service" not in function
    assert "mlashmcomplex" not in function
    assert " rctd " not in function
    assert "simaai-pipeline-manager.service" in function
    assert "simaai_pipeline_handler_new" in function


def test_native_video_sender_kind_contract(tmp_path):
    binary = tmp_path / "video-sender-kind"
    subprocess.run([
        "/usr/bin/g++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
        "-DSIMA_NEAT_INTERNAL=1", "-I", str(ROOT / "src"), "-I", str(ROOT / "include"),
        str(ROOT / "tests/unit_testing/unit_video_sender_kind_contract_test.cpp"),
        str(ROOT / "src/nodes/groups/VideoSenderRawIngress.cpp"),
        str(ROOT / "src/builder/OutputSpec.cpp"), "-o", str(binary),
    ], check=True, timeout=90)
    kind = subprocess.check_output(["file", str(binary)], text=True)
    assert "x86-64" in kind and "aarch64" not in kind, kind
    subprocess.run([str(binary)], check=True, timeout=10)
