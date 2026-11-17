"""Exercise dependency verification and argument rejection without host mutations."""
import os
from pathlib import Path
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def build_env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scripts = {
        "apt-get": 'echo "$*" >> "$APT_LOG"',
        "apt-cache": 'exit 1',
        "sudo": 'exec "$@"',
        "dpkg-query": 'if [[ "${@: -1}" = "${MISSING_PACKAGE:-}" ]]; then exit 1; fi; printf "install ok installed"',
    }
    for name, script in scripts.items():
        path = bin_dir / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + script + "\n")
        path.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", APT_LOG=str(tmp_path / "apt.log"))
    for name in ("CMAKE_BUILD_PARALLEL_LEVEL", "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
                 "CMAKE_TOOLCHAIN_FILE", "PKG_CONFIG_SYSROOT_DIR", "PKG_CONFIG_LIBDIR", "PKG_CONFIG_PATH",
                 "CMAKE_LIBRARY_PATH", "CMAKE_INCLUDE_PATH", "CMAKE_PREFIX_PATH", "CMAKE_FIND_ROOT_PATH",
                 "LIBRARY_PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
        env.pop(name, None)
    return env


def run_build(env, *args):
    # Supply an Ubuntu identity without changing the test host's /etc/os-release.
    script = '''
source() {
  if [[ "$1" == /etc/os-release ]]; then
    ID=ubuntu
    VERSION_ID=22.04
  else
    builtin source "$@"
  fi
}
export -f source
exec bash "$@"
'''
    return subprocess.run(["bash", "-c", script, "--", str(ROOT / "pcie_host/build.sh"), *args],
                          env=env, text=True, capture_output=True)


def test_preinstalled_dependencies_never_run_apt(build_env):
    result = run_build(build_env, "--all", "--skip-deps", "--install-deps-only", "--jobs", "8")
    assert result.returncode == 0, result.stderr
    assert not Path(build_env["APT_LOG"]).exists()


def test_incomplete_image_fails_without_apt_fallback(build_env):
    build_env["MISSING_PACKAGE"] = "libopencv-dev"
    result = run_build(build_env, "--all", "--skip-deps", "--install-deps-only")
    assert result.returncode != 0
    assert "requires installed package libopencv-dev" in result.stderr
    assert not Path(build_env["APT_LOG"]).exists()


def test_default_still_installs_dependencies(build_env):
    result = run_build(build_env, "--all", "--install-deps-only")
    assert result.returncode == 0, result.stderr
    calls = Path(build_env["APT_LOG"]).read_text().splitlines()
    assert calls[0] == "update"
    assert "libopencv-dev" in calls[1]
    assert "python3-venv" in calls[1]


@pytest.mark.parametrize("jobs", ["0", "-1", "1.5", "abc", ""])
def test_invalid_jobs_fail_before_package_changes(build_env, jobs):
    result = run_build(build_env, "--jobs", jobs, "--install-deps-only")
    assert result.returncode == 2
    assert "positive integer" in result.stderr
    assert not Path(build_env["APT_LOG"]).exists()


def test_cli_jobs_override_invalid_environment(build_env):
    build_env["CMAKE_BUILD_PARALLEL_LEVEL"] = "invalid"
    result = run_build(build_env, "--skip-deps", "--install-deps-only")
    assert result.returncode == 2
    result = run_build(build_env, "--skip-deps", "--install-deps-only", "--jobs", "4")
    assert result.returncode == 0, result.stderr


def test_ci_keeps_all_native_targets_and_validation():
    workflow = yaml.safe_load((ROOT / ".github/workflows/vulcan-ci.yml").read_text())
    job = workflow["jobs"]["build-pciehost-packages"]
    assert {(row["distro"], row["deb_arch"], row["ubuntu_version"])
            for row in job["strategy"]["matrix"]["include"]} == {
                (distro, arch, version) for distro, version in [("ubuntu22", "22.04"), ("ubuntu24", "24.04")]
                for arch in ["amd64", "arm64"]}
    assert "${{ matrix.deb_arch }}" in job["runs-on"]
    assert "ubuntu" in job["runs-on"]
    steps = {step.get("name"): step for step in job["steps"]}
    assert "Run PCIe host unit tests" in steps
    assert "Smoke test staged PCIe host packages and wheel" in steps
    assert '--skip-deps --jobs "${PCIE_HOST_BUILD_JOBS:-$(nproc)}"' in steps["Build PCIe host Debian packages"]["run"]


def test_shared_image_qualification_preserves_all_targets_without_publication():
    regular = yaml.safe_load((ROOT / ".github/workflows/vulcan-ci.yml").read_text())["jobs"]["build-pciehost-packages"]
    workflow = yaml.safe_load((ROOT / ".github/workflows/qualify-ubuntu-build-images.yml").read_text())
    job = workflow["jobs"]["qualify"]
    assert {(row["distro"], row["arch"]) for row in job["strategy"]["matrix"]["include"]} == {
        (row["distro"], row["deb_arch"]) for row in regular["strategy"]["matrix"]["include"]}
    assert "069171140053.dkr.ecr.us-west-2.amazonaws.com/vulcan/ubuntu-build:" in job["container"]["image"]
    assert "-media-" in job["container"]["image"]
    assert "069171140053.dkr.ecr.us-west-2.amazonaws.com/vulcan/ubuntu-build:" in regular["container"]["image"]
    assert "-media-" in regular["container"]["image"]
    assert "id-token" not in workflow["permissions"]
    assert "credentials" not in job["container"]
    assert "credentials" not in regular["container"]
    scripts = "\n".join(step.get("run", "") for step in job["steps"])
    assert "validate_build_image.sh" in scripts
    assert "aws s3" not in scripts
    assert "CORE_VALIDATION_SHA" in next(step["env"] for step in job["steps"] if step.get("name") == "Build and test PCIe host consumer")
