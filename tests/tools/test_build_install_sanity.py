"""Exercise actual install sanity and installed test-asset discovery."""
from pathlib import Path
import io
import os
import platform
import shutil
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _fixture(tmp_path: Path, libdir: str, defect: str = "") -> Path:
    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    for name in ("libsima_neat.so.3.0.0", "libsima_neat.a", "neat.h"):
        (source / name).write_text("install-layout fixture; not executable\n")
    (source / "SimaNeatConfig.cmake").write_text('set(SimaNeat_ABI_VERSION "4")\n')
    (source / "libsima_neat.so.4").symlink_to("libsima_neat.so.3.0.0")
    (source / "libsima_neat.so").symlink_to(
        "libsima_neat.so.3" if defect == "wrong-abi" else "libsima_neat.so.4"
    )
    core_files = "libsima_neat.so.4"
    if defect != "missing-runtime":
        core_files += " libsima_neat.so.3.0.0"
    if defect == "leaked-dev-file":
        core_files += " libsima_neat.a"
    (source / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.16)
project(InstallSanityFixture NONE)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE PATH "Library directory")
configure_file(SimaNeatConfig.cmake SimaNeatConfig.cmake COPYONLY)
install(FILES {core_files} DESTINATION "${{CMAKE_INSTALL_LIBDIR}}" COMPONENT core)
install(FILES libsima_neat.a libsima_neat.so DESTINATION "${{CMAKE_INSTALL_LIBDIR}}" COMPONENT dev)
install(FILES neat.h DESTINATION include COMPONENT dev)
install(FILES SimaNeatConfig.cmake DESTINATION "${{CMAKE_INSTALL_LIBDIR}}/cmake/SimaNeat" COMPONENT dev)
''')
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), f"-DCMAKE_INSTALL_LIBDIR:PATH={libdir}"],
        check=True, capture_output=True, text=True,
    )
    return build


def _run_sanity(tmp_path: Path, build: Path) -> subprocess.CompletedProcess:
    # Invoke the production function, not a copied/reimplemented test oracle.
    script = (ROOT / "build.sh").read_text()
    start = script.index("run_install_sanity_check() {")
    end = script.index("\nbuild_deb_if_requested() {", start)
    env = dict(os.environ, BUILD_DIR=str(build), TMPDIR=str(tmp_path))
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + script[start:end] + "\nrun_install_sanity_check\n"],
        env=env, capture_output=True, text=True,
    )


@pytest.mark.parametrize("libdir", ["lib", "lib64", "lib/aarch64-linux-gnu"])
def test_supported_library_layouts(tmp_path, libdir):
    result = _run_sanity(tmp_path, _fixture(tmp_path, libdir))
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(tmp_path.glob("sima-neat-install-test.*")), "successful staging must be cleaned"


@pytest.mark.parametrize("defect,diagnostic", [
    ("missing-runtime", "versioned libsima_neat shared object missing"),
    ("wrong-abi", "expected ABI 4"),
    ("leaked-dev-file", "core install tree contains development files"),
])
def test_multiarch_checks_still_reject_bad_packages(tmp_path, defect, diagnostic):
    result = _run_sanity(tmp_path, _fixture(tmp_path, "lib/aarch64-linux-gnu", defect))
    assert result.returncode != 0
    assert diagnostic in result.stdout + result.stderr
    assert list(tmp_path.glob("sima-neat-install-test.*")), "failed evidence must be preserved"


@pytest.mark.parametrize("libdir", ["/outside/lib", "../outside", "lib/../../outside"])
def test_escaping_library_directory_rejected_before_install(tmp_path, libdir):
    result = _run_sanity(tmp_path, _fixture(tmp_path, libdir))
    assert result.returncode != 0
    assert "requires a relative CMAKE_INSTALL_LIBDIR" in result.stderr
    assert not list(tmp_path.glob("sima-neat-install-test.*"))


@pytest.fixture(scope="module")
def lifecycle_resolver(tmp_path_factory):
    # Compile the actual cold fixture resolver, without Core/device execution.
    source = (ROOT / "tests/stress/stress_model_lifecycle_test.cpp").read_text()
    start = source.index("std::string resolve_model_pack_for_stress() {")
    end = source.index("\nstd::vector<int64_t> contiguous_strides_bytes", start)
    build = tmp_path_factory.mktemp("lifecycle-resolver")
    probe = build / "probe.cpp"
    probe.write_text('#include "asset_utils.h"\n#include <iostream>\n' + source[start:end] +
                     '\nint main() { std::cout << resolve_model_pack_for_stress(); }\n')
    binary = build / "probe"
    result = subprocess.run(
        ["c++", "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "tests"),
         str(probe), "-o", str(binary)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("archive", ["resnet_50_mpk.tar.gz", "yolo_v8s_mpk.tar.gz"])
@pytest.mark.parametrize("working_directory", ["tests", "tests/nested"])
@pytest.mark.parametrize("override", ["", "generic", "stress", "missing"])
def test_installed_lifecycle_uses_shared_model_root(
    tmp_path, lifecycle_resolver, archive, working_directory, override,
):
    installed = tmp_path / "extras/lib/sima-neat"
    cwd = installed / working_directory
    cwd.mkdir(parents=True)
    manifest = installed / "test-fixtures/runtime_manifest.json"
    manifest.parent.mkdir()
    # Use the producer's installed manifest, not a test-only path convention.
    cmake = (ROOT / "CMakeLists.txt").read_text()
    content = cmake.split("set(SIMANEAT_TEST_RUNTIME_MANIFEST_INSTALLED_CONTENT [=[\n", 1)[1]
    manifest.write_text(content.split("\n]=])", 1)[0])
    cached = installed / "tmp" / archive
    cached.parent.mkdir()
    with tarfile.open(cached, "w:gz") as stream:
        member = tarfile.TarInfo("fixture/mpk.json")
        member.size = 2
        stream.addfile(member, io.BytesIO(b"{}"))

    home, temporary, bin_dir = (tmp_path / name for name in ("home", "temporary", "bin"))
    for directory in (home, temporary, bin_dir):
        directory.mkdir()
    unexpected_download = tmp_path / "unexpected-download"
    cli = bin_dir / "sima-cli"
    cli.write_text(f"#!/bin/sh\ntouch '{unexpected_download}'\nexit 99\n")
    cli.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key not in {"SIMA_STRESS_MODEL_TAR", "SIMA_MODEL_TAR", "SIMA_RESNET50_TAR", "SIMA_YOLO_TAR"}}
    env.update(HOME=str(home), TMPDIR=str(temporary), PATH=f"{bin_dir}:{os.environ['PATH']}")
    expected = cached
    if override in {"generic", "stress"}:
        expected = tmp_path / "explicit-model.tar.gz"
        shutil.copyfile(cached, expected)
        env["SIMA_MODEL_TAR"] = str(expected)
    if override == "stress":
        expected = tmp_path / "explicit-stress-model.tar.gz"
        shutil.copyfile(cached, expected)
        env["SIMA_STRESS_MODEL_TAR"] = str(expected)
    elif override == "missing":
        env.update(SIMA_MODEL_TAR=str(tmp_path / "missing-generic"),
                   SIMA_STRESS_MODEL_TAR=str(tmp_path / "missing-stress"))

    description = subprocess.check_output(["file", str(lifecycle_resolver)], text=True)
    architecture = {"x86_64": "x86-64", "amd64": "x86-64", "aarch64": "ARM aarch64", "arm64": "ARM aarch64"}
    assert architecture.get(platform.machine().lower(), "unknown native architecture") in description
    result = subprocess.run([str(lifecycle_resolver)], cwd=cwd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == str(expected)
    assert cached.is_file(), "discovery must retain the shared fixture"
    assert not unexpected_download.exists(), "local lifecycle discovery must not acquire models"
