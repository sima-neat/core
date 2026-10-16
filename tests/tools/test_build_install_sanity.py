"""Exercise the producer's actual install-sanity function with real CMake installs."""
from pathlib import Path
import os
import subprocess

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
