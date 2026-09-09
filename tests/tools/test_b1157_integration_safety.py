"""Host-only B1157 pairing and recovery regressions; never touch a DevKit."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "deps/manifest.json").read_text())
PROFILE = {
    "runtime_profile": MANIFEST["runtime-profile"],
    "kernel_commit": MANIFEST["kernel-commit"],
    "sysroot_version": MANIFEST["expected-internals-sysroot"],
}


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


@pytest.mark.parametrize("owner", ["sima-neat", "sima-neat-internals"])
@pytest.mark.parametrize("marker", ["valid", "malformed", "dangling"])
def test_shell_refuses_all_recovery_mutation_with_direct_marker(tmp_path, owner, marker):
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc/buildinfo").write_text("MACHINE=modalix\nDISTRO_VERSION=2.1.3\n")
    receipt = tmp_path / "usr/share" / owner / "runtime-profile.json"
    receipt.parent.mkdir(parents=True)
    if marker == "dangling":
        receipt.symlink_to("missing")
    else:
        receipt.write_text(json.dumps(PROFILE) if marker == "valid" else "{")
    result = bash(f"""
export NEAT_RECOVERY_FUNCTIONS_ONLY=ON
export SIMA_NEAT_RECOVERY_HARD_RESET=1 SIMA_NEAT_RECOVERY_ALLOW_UNSAFE_RESET=1
source {shlex.quote(str(ROOT / 'scripts/fix_devkit_runtime.sh'))}
legacy_runtime_recovery_allowed() {{ legacy_runtime_recovery_allowed_at {shlex.quote(str(tmp_path))}; }}
run_step() {{ echo MUTATED; }}
quarantine_stale_global_dispatcher_libs() {{ echo MUTATED; }}
activate_staged_ev74_firmware_if_needed() {{ echo MUTATED; }}
recover_devkit_runtime_main
""")
    assert result.returncode != 0
    assert "MUTATED" not in result.stdout
    assert "Refusing legacy recovery" in result.stderr


def artifact_identity():
    return {
        "platform-version": MANIFEST["platform-version"],
        "runtime-profile": PROFILE["runtime_profile"],
        "kernel-commit": PROFILE["kernel_commit"],
        "sysroot-version": PROFILE["sysroot_version"],
    }


@pytest.mark.parametrize("mismatch", [None, "platform-version", "runtime-profile", "kernel-commit", "sysroot-version"])
def test_artifact_profile_is_exact(tmp_path, mismatch):
    artifact = artifact_identity()
    if mismatch:
        artifact[mismatch] = "old"
    receipt = tmp_path / "internals-manifest.json"
    receipt.write_text(json.dumps(artifact))
    result = subprocess.run([
        "python3", str(ROOT / "scripts/build/validate_internals_profile.py"),
        str(ROOT / "deps/manifest.json"), str(receipt),
    ], text=True, capture_output=True, check=False)
    assert (result.returncode == 0) == (mismatch is None), result.stderr


def test_cmake_profile_rejects_old_or_incomplete_installed_package(tmp_path):
    template = (ROOT / "cmake/RequireInternalsProfile.cmake.in").read_text()
    for key, value in {
        "SIMANEAT_RUNTIME_PROFILE": PROFILE["runtime_profile"],
        "SIMANEAT_KERNEL_COMMIT": PROFILE["kernel_commit"],
        "SIMANEAT_INTERNALS_SYSROOT": PROFILE["sysroot_version"],
    }.items():
        template = template.replace(f"@{key}@", value)
    for field in (None, "RUNTIME_PROFILE", "KERNEL_COMMIT", "SYSROOT_VERSION"):
        variables = {
            "RUNTIME_PROFILE": PROFILE["runtime_profile"],
            "KERNEL_COMMIT": PROFILE["kernel_commit"],
            "SYSROOT_VERSION": PROFILE["sysroot_version"],
        }
        if field:
            variables.pop(field)
        config = tmp_path / "profile.cmake"
        config.write_text("\n".join(f'set(NeatInternals_{k} "{v}")' for k, v in variables.items()) + "\n" + template)
        result = subprocess.run(["cmake", "-P", str(config)], text=True, capture_output=True)
        assert (result.returncode == 0) == (field is None), result.stderr


def make_deb(directory, name, version, receipt=None):
    package = directory / name
    control = package / "DEBIAN/control"
    control.parent.mkdir(parents=True)
    control.write_text(f"Package: {name}\nVersion: {version}\nArchitecture: all\nMaintainer: Test <test@example.invalid>\nDescription: host fixture\n")
    if receipt is not None:
        path = package / "usr/share/sima-neat-internals/runtime-profile.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(receipt))
    output = directory / f"{name}.deb"
    subprocess.run(["dpkg-deb", "--build", "--root-owner-group", str(package), str(output)], check=True, capture_output=True)
    return output


@pytest.mark.parametrize("mode", ["matched", "missing", "old_profile", "mixed_version"])
def test_installer_checks_package_payload_even_with_platform_override(tmp_path, mode):
    receipt = dict(PROFILE)
    if mode == "old_profile":
        receipt["runtime_profile"] = "legacy"
    runtime = make_deb(tmp_path, "neat-runtime", "1.0", None if mode == "missing" else receipt)
    plugins = make_deb(tmp_path, "neat-gst-plugins", "2.0" if mode == "mixed_version" else "1.0")
    result = bash(f"""
source {shlex.quote(str(ROOT / 'tools/install_neat_framework.sh'))}
NEAT_PACKAGE_MANIFEST={shlex.quote(str(ROOT / 'deps/manifest.json'))}
NEAT_INSTALLER_SKIP_PLATFORM_CHECK=ON
DEBS=({shlex.quote(str(runtime))} {shlex.quote(str(plugins))})
validate_bundled_internals_profile
""")
    assert (result.returncode == 0) == (mode == "matched"), result.stderr


def test_manifest_disallows_snap_and_gates_before_dependency_install():
    assert MANIFEST["platform-version"] == "3.0.0"
    assert MANIFEST["internals"] == {"branch": "codex/b1157-internals-compat", "spec": "latest"}
    script = (ROOT / "build.sh").read_text()
    function = script.split("ensure_neat_internals() {", 1)[1].split("\n}\n", 1)[0]
    assert function.index("validate_internals_runtime_profile") < function.index("sync_sysroot_from_internals_manifest")
    assert function.index("validate_internals_runtime_profile") < function.index("collect_plugin_files_from_debs")
    assert "Direct-driver Core requires an explicit Internals ref" in script
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert '"neat-runtime-profile-b1157 (= 1)"' in cmake


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


def test_b1157_git_sysroot_receipt_updates_exactly():
    from test_internals_package_boundary import run_sync
    result, calls = run_sync({"sysroot-version": PROFILE["sysroot_version"]}, "3.0.0")
    assert result.returncode == 0, result.stderr
    assert calls == [f"update {PROFILE['sysroot_version']}", "status"]


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
