"""Exercise package metadata without importing or executing a target extension."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/build/generate_package_buildinfo.py"


@pytest.fixture
def selected(tmp_path):
    repo = tmp_path / "core"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.9"\n')
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "metadata fixture"], check=True)
    build = tmp_path / "build"
    build.mkdir()
    sysroot = tmp_path / "target"
    include = sysroot / "usr/include/python3.13"
    include.mkdir(parents=True)
    (include / "Python.h").write_text('#include "patchlevel.h"\n#include "pyconfig.h"\n')
    (include / "patchlevel.h").write_text("#define PY_MAJOR_VERSION 3\n#define PY_MINOR_VERSION 13\n")
    (include / "pyconfig.h").write_text("/* No debug or free-threaded ABI flags. */\n")
    extension = build / "python/_pyneat_core.cpython-313-x86_64-linux-gnu.so"
    extension.parent.mkdir()
    extension.write_text("Metadata fixture only; never loaded or executed.\n")
    compiler = Path(shutil.which("c++")).resolve()
    cmake_compiler = build / "CMakeFiles/3.28.3/CMakeCXXCompiler.cmake"
    cmake_compiler.parent.mkdir(parents=True)
    cmake_compiler.write_text(f'set(CMAKE_CXX_COMPILER "{compiler}")\n'
                              'set(CMAKE_CXX_COMPILER_ID "GNU")\n'
                              'set(CMAKE_CXX_COMPILER_VERSION "14.2.0")\n')
    cache = build / "CMakeCache.txt"
    cache.write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={repo}\n'
                     f'CMAKE_SYSROOT:PATH={sysroot}\n'
                     f'Python_INCLUDE_DIR:PATH={include}\n'
                     'Python_EXECUTABLE:FILEPATH=/unrelated/host/python3.12\n'
                     '_Python_INTERPRETER_PROPERTIES:INTERNAL=Python;3;12;3\n'
                     'SIMANEAT_BUILD_PYTHON:BOOL=ON\n'
                     'PYNEAT_EXT_SUFFIX:STRING=.cpython-313-x86_64-linux-gnu.so\n'
                     'CMAKE_CACHE_MAJOR_VERSION:INTERNAL=3\n'
                     'CMAKE_CACHE_MINOR_VERSION:INTERNAL=28\n'
                     'CMAKE_CACHE_PATCH_VERSION:INTERNAL=3\n')
    host_release = tmp_path / "host-sdk-release"
    host_release.write_text("SDK Version=2.1.2_container\neLXr Version=2.1.2_container\n")
    return dict(repo=repo, build=build, sysroot=sysroot, include=include, extension=extension,
                cache=cache, host_release=host_release, compiler=compiler, cmake_compiler=cmake_compiler,
                output=tmp_path / "buildinfo.json")


def invoke(f, *args, final=True, env=None):
    command = [sys.executable, str(GENERATOR), "--repo-root", str(f["repo"]),
               "--output", str(f["output"]), "--package-name", "sima-neat",
               "--package-version", "0.6.0+test", "--platform-version", "3.0.0+cohort",
               "--sdk-release-file", str(f["host_release"])]
    if final:
        command += ["--phase", "final", "--cmake-cache", str(f["cache"]),
                    "--python-extension", str(f["extension"])]
    command += list(map(str, args))
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("GITHUB_")}
    return subprocess.run(command, env={**clean_env, **(env or {})}, capture_output=True, text=True)


def success(f, *args, **kwargs):
    result = invoke(f, *args, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(f["output"].read_text())


def test_actual_target_abi_not_host_interpreter_or_host_sdk(selected):
    data = success(selected)
    compat = data["compatibility"]
    assert data["metadata_phase"] == "final"
    assert compat["pyneat"]["python_abi"] == "CPython 3.13 (cp313)"
    assert compat["pyneat"]["extension_platform"] == "x86_64-linux-gnu"
    assert compat["pyneat"]["extension_sha256"]
    assert compat["host_sdk"]["sdk"] == "2.1.2"
    assert compat["target"]["platform_sw"] == "3.0.0"
    assert compat["target_sdk"]["status"] == "unavailable"
    assert "sdk" not in compat["target_sdk"]
    assert "2.1.2" not in json.dumps(compat["target_sdk"])
    assert compat["target_sdk"]["compiler"]["path"] == str(selected["compiler"])
    assert compat["target_sdk"]["compiler"]["version"] == "14.2.0"
    assert data["build"]["git_dirty"] is False
    assert data["source"] == "local"


def test_initial_metadata_is_explicitly_provisional_and_has_no_fabricated_defaults(selected):
    selected["host_release"].unlink()
    data = success(selected, final=False)
    assert data["metadata_phase"] == "provisional"
    for key in ("pyneat", "host_sdk", "target_sdk"):
        assert data["compatibility"][key]["status"] == "unavailable"
    assert "python_abi" not in data["compatibility"]["pyneat"]
    assert "toolchain" not in data["compatibility"]["host_sdk"]
    assert "sysroot" not in data["compatibility"]["target_sdk"]


@pytest.mark.parametrize("flags", ["d", "t", "td"])
def test_abi_flags_retained_and_verified(selected, flags):
    f = selected
    (f["include"] / "pyconfig.h").write_text(
        ("#define Py_DEBUG 1\n" if "d" in flags else "") +
        ("#define Py_GIL_DISABLED 1\n" if "t" in flags else ""))
    extension = f["extension"].with_name(f"_pyneat_core.cpython-313{flags}-x86_64-linux-gnu.so")
    f["extension"].rename(extension)
    f["extension"] = extension
    f["cache"].write_text(f["cache"].read_text().replace("cpython-313-", f"cpython-313{flags}-"))
    assert success(f)["compatibility"]["pyneat"]["abi_tag"] == "cp313" + flags


def test_configured_compiler_flags_participate(selected):
    with selected["cache"].open("a") as out:
        out.write("CMAKE_CXX_FLAGS:STRING=-DPy_DEBUG=1\n")
    result = invoke(selected)
    assert result.returncode != 0
    assert "contradicts target Python headers" in result.stderr


@pytest.mark.parametrize("defect, diagnostic", [
    ("missing-cache", "No such file"),
    ("bad-cache", "does not belong"),
    ("other-source", "does not belong"),
    ("missing-compiler-metadata", "No such file"),
    ("compiler-conflict", "compiler override contradicts"),
    ("missing-headers", "does not exist"),
    ("missing-Python.h", "Missing target Python header"),
    ("bad-version", "PY_MAJOR_VERSION"),
    ("version-mismatch", "contradicts target Python headers"),
    ("debug-mismatch", "contradicts target Python headers"),
    ("suffix-mismatch", "contradicts selected CMake suffix"),
    ("missing-extension", "missing or empty"),
    ("empty-extension", "missing or empty"),
    ("outside-extension", "outside selected CMake build"),
])
def test_invalid_selected_evidence_fails_closed(selected, tmp_path, defect, diagnostic):
    f = selected
    if defect == "missing-cache":
        f["cache"].unlink()
    elif defect == "bad-cache":
        f["cache"].write_text("not a CMake cache\n")
    elif defect == "other-source":
        f["cache"].write_text(f["cache"].read_text().replace(str(f["repo"]), str(tmp_path)))
    elif defect == "missing-compiler-metadata":
        f["cmake_compiler"].unlink()
    elif defect == "compiler-conflict":
        with f["cache"].open("a") as out:
            out.write("CMAKE_CXX_COMPILER:FILEPATH=/unrelated/compiler\n")
    elif defect == "missing-headers":
        shutil.rmtree(f["include"])
    elif defect == "missing-Python.h":
        (f["include"] / "Python.h").unlink()
    elif defect == "bad-version":
        (f["include"] / "patchlevel.h").write_text("/* unrelated header */\n")
    elif defect == "version-mismatch":
        (f["include"] / "patchlevel.h").write_text("#define PY_MAJOR_VERSION 3\n#define PY_MINOR_VERSION 12\n")
    elif defect == "debug-mismatch":
        (f["include"] / "pyconfig.h").write_text("#define Py_DEBUG 1\n")
    elif defect == "suffix-mismatch":
        f["cache"].write_text(f["cache"].read_text().replace("cpython-313-", "cpython-312-"))
    elif defect == "missing-extension":
        f["extension"].unlink()
    elif defect == "empty-extension":
        f["extension"].write_bytes(b"")
    elif defect == "outside-extension":
        path = tmp_path / f["extension"].name
        f["extension"].rename(path)
        f["extension"] = path
    result = invoke(f)
    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert not f["output"].exists()


@pytest.mark.parametrize("option,key", [("--sysroot", "sysroot"), ("--python-include-dir", "include")])
def test_explicit_selection_conflicts_rejected(selected, tmp_path, option, key):
    result = invoke(selected, option, tmp_path)
    assert result.returncode != 0
    assert "override contradicts" in result.stderr
    success(selected, option, selected[key])


def test_target_sdk_override_and_host_override_are_separate(selected, tmp_path):
    target = tmp_path / "target-sdk-release"
    target.write_text("SDK Version=9.8.7_target\neLXr Version=6.5.4_target\n")
    compat = success(selected, "--target-sdk-release-file", target,
                     "--toolchain", "explicit caller label")["compatibility"]
    assert compat["target_sdk"]["sdk"] == "9.8.7"
    assert compat["target_sdk"]["elxr"] == "6.5.4"
    assert compat["target_sdk"]["toolchain_override"] == "explicit caller label"
    assert compat["target_sdk"]["compiler"]["version"] == "14.2.0"
    assert compat["host_sdk"]["sdk"] == "2.1.2"
    actual = selected["sysroot"] / "etc/sdk-release"
    actual.parent.mkdir()
    actual.write_text("SDK Version=9.8.7_other_cohort\neLXr Version=6.5.4_target\n")
    result = invoke(selected, "--target-sdk-release-file", target)
    assert result.returncode != 0
    assert "contradicts selected sysroot" in result.stderr
    target.unlink()
    assert "does not exist" in invoke(selected, "--target-sdk-release-file", target).stderr


def test_python_disabled_does_not_consume_stale_python_cache_entries(selected):
    selected["cache"].write_text(selected["cache"].read_text().replace("SIMANEAT_BUILD_PYTHON:BOOL=ON", "SIMANEAT_BUILD_PYTHON:BOOL=OFF"))
    shutil.rmtree(selected["include"])
    data = success(selected, "--phase", "final", "--cmake-cache", selected["cache"],
                   "--python-enabled", "off", final=False)
    assert data["compatibility"]["pyneat"]["status"] == "disabled"
    assert "python_abi" not in data["compatibility"]["pyneat"]
    assert "Python-disabled" in invoke(selected).stderr
    assert "contradicts selected CMake cache" in invoke(selected, "--python-enabled", "on").stderr


def test_final_requires_cache_and_extension(selected):
    assert "requires --cmake-cache" in invoke(selected, "--phase", "final", final=False).stderr
    result = invoke(selected, "--phase", "final", "--cmake-cache", selected["cache"], final=False)
    assert result.returncode != 0
    assert "requires selected headers and built extension" in result.stderr


def test_dirty_checkout_not_reported_as_clean_ci_sha(selected):
    (selected["repo"] / "pyproject.toml").write_text("# local changed source\n")
    (selected["repo"] / "untracked.py").write_text("# local addition\n")
    data = success(selected, env={"GITHUB_SHA": "f" * 40, "GITHUB_RUN_ID": "123"})
    assert data["build"]["git_sha"] != "f" * 40
    assert data["build"]["ci_requested_git_sha"] == "f" * 40
    assert data["build"]["git_dirty"] is True
    assert len(data["build"]["git_status_porcelain"]) == 2
    assert data["build"]["tracked_diff_sha256"]
    assert data["source"] == "local"


def run_shell_generator(f, phase):
    script = (ROOT / "build.sh").read_text()
    start = script.index("generate_package_buildinfo_json() {")
    end = script.index("\nsync_compatibility_snapshot() {", start)
    generator = f["repo"] / "scripts/build/generate_package_buildinfo.py"
    generator.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(GENERATOR, generator)
    backend = f["repo"] / "backend"
    backend.mkdir(exist_ok=True)
    (backend / "resolved_manifest.json").write_text('{"fixture": "actual selected receipt passthrough"}\n')
    env = dict(os.environ, REPO_ROOT=str(f["repo"]), BUILD_DIR=str(f["build"]),
               NEAT_PACKAGE_BUILDINFO_JSON=str(f["output"]), BUILD_PYTHON="ON", ELXR_SDK="ON",
               SYSROOT=str(f["sysroot"]), ELXR_TARGET_PYTHON_INCLUDE_DIR=str(f["include"]),
               ELXR_SDK_RELEASE_FILE=str(f["host_release"]), NEAT_PACKAGE_NAME="sima-neat",
               NEAT_VULCAN_ENV="dev", NEAT_LLIMA_RESOLVED_REF="main:abc", NEAT_BACKEND_DIR=str(backend))
    text = ('set -euo pipefail\ncompute_neat_package_version() { echo 0.6.0; }\n'
            'compute_neat_platform_version() { echo 3.0.0; }\n' + script[start:end] +
            f'\ngenerate_package_buildinfo_json {phase}\necho CONSUMER_REACHED\n')
    return subprocess.run(["bash", "-c", text], cwd=f["repo"], env=env, text=True, capture_output=True)


def test_actual_shell_two_phase_generation_and_receipt_preservation(selected):
    result = run_shell_generator(selected, "provisional")
    assert result.returncode == 0, result.stderr
    assert json.loads(selected["output"].read_text())["metadata_phase"] == "provisional"
    result = run_shell_generator(selected, "final")
    assert result.returncode == 0, result.stderr
    data = json.loads(selected["output"].read_text())
    assert data["metadata_phase"] == "final"
    assert data["compatibility"]["pyneat"]["abi_tag"] == "cp313"
    assert data["native_dependencies"]["fixture"] == "actual selected receipt passthrough"
    assert data["dependencies"]["llima"] == {"ref": "main", "spec": "abc"}


@pytest.mark.parametrize("defect", ["missing-extension", "multiple-extensions", "invalid-cache"])
def test_actual_shell_stops_before_consumers_on_bad_final_evidence(selected, defect):
    if defect == "missing-extension":
        selected["extension"].unlink()
    elif defect == "multiple-extensions":
        selected["extension"].with_name("_pyneat_core.other.so").write_text("stale fixture")
    else:
        selected["cache"].write_text("invalid cache\n")
    result = run_shell_generator(selected, "final")
    assert result.returncode != 0
    assert "CONSUMER_REACHED" not in result.stdout


def test_canonical_main_refreshes_before_install_or_package_consumers():
    script = (ROOT / "build.sh").read_text()
    main = script[script.index("main() {"):]
    provisional = main.index("  generate_package_buildinfo_json provisional\n")
    configure = main.index("  configure_cmake\n")
    build = main.index("  build_targets\n")
    final = main.index("  generate_package_buildinfo_json final\n")
    assert provisional < configure < build < final
    for consumer in ("build_docs_if_requested", "run_install_sanity_check", "build_deb_if_requested",
                     "build_extras_archive_if_requested", "build_python_wheel_if_requested",
                     "install_artifacts_into_current_environment_if_requested", "deploy_artifacts_to_devkit_if_requested"):
        assert final < main.index(f"  {consumer}\n")


def test_untracked_content_and_symlinks_have_deterministic_identity(selected):
    extra = selected["repo"] / "new.cpp"
    extra.write_text("first content\n")
    link = selected["repo"] / "new-link"
    link.symlink_to("new.cpp")
    first = success(selected)["build"]
    assert first["untracked_files"]["new-link"] == {"type": "symlink", "target": "new.cpp"}
    assert first["untracked_files"]["new.cpp"]["sha256"]
    assert success(selected)["build"]["untracked_files_sha256"] == first["untracked_files_sha256"]
    extra.write_text("different content\n")
    assert success(selected)["build"]["untracked_files_sha256"] != first["untracked_files_sha256"]
    selected["output"] = selected["repo"] / "buildinfo.json"
    second = success(selected)["build"]
    third = success(selected)["build"]
    assert second["untracked_files_sha256"] == third["untracked_files_sha256"]
    assert third["untracked_identity_exclusions"] == ["buildinfo.json"]
    assert "buildinfo.json" not in third["untracked_files"]


@pytest.mark.parametrize("tag, valid", [("cp313-cp313", True), ("cp311-cp311", False)])
def test_actual_wheel_install_message_uses_verified_target_abi(selected, tmp_path, tag, valid):
    success(selected)
    dist = tmp_path / "dist"
    dist.mkdir()
    metadata = dist / "metadata-pyneat.json"
    metadata.write_text("{}\n")
    script = (ROOT / "build.sh").read_text()
    start = script.index('  WHEEL_BASENAME="$(basename "${wheel_path}")"')
    start = script.index("\n", start) + 1
    end = script.index("\nPY\n", start)
    result = subprocess.run([sys.executable, "-c", script[start:end]], cwd=tmp_path,
                            env=dict(os.environ, WHEEL_BASENAME=f"pyneat-0.6.0-{tag}-linux_x86_64.whl",
                                     NEAT_BUILDINFO_JSON=str(selected["output"])), text=True, capture_output=True)
    if valid:
        assert result.returncode == 0, result.stderr
        message = json.loads(metadata.read_text())["installation"]["post-message"]
        assert "CPython 3.13 (cp313) virtual environment" in message
        assert "Python 3.11" not in message
    else:
        assert result.returncode != 0
        assert "contradict final selected-target ABI" in result.stderr
        assert json.loads(metadata.read_text()) == {}


@pytest.mark.parametrize("flag", ["--sysroot={}", "--sysroot {}", "-isysroot {}", "-isysroot{}", "-isysroot={}"])
def test_contradictory_compile_flag_sysroot_rejected(selected, tmp_path, flag):
    with selected["cache"].open("a") as out:
        out.write(f"CMAKE_CXX_FLAGS:STRING={flag.format(tmp_path)}\n")
    result = invoke(selected)
    assert result.returncode != 0
    assert "flags contradict configured compile sysroot" in result.stderr


def test_matching_repeated_sysroot_flags_are_accepted(selected):
    with selected["cache"].open("a") as out:
        out.write(f"CMAKE_CXX_FLAGS:STRING=--sysroot={selected['sysroot']} --sysroot {selected['sysroot']}\n")
    assert success(selected)["compatibility"]["target_sdk"]["sysroot"] == str(selected["sysroot"])


@pytest.mark.parametrize("authority", ["CMAKE_SYSROOT_COMPILE:PATH=", "CMAKE_CXX_FLAGS:STRING=--sysroot="])
def test_compile_only_sysroot_is_authority(selected, authority):
    selected["cache"].write_text(selected["cache"].read_text().replace("CMAKE_SYSROOT:PATH=", authority))
    assert success(selected)["compatibility"]["target_sdk"]["sysroot"] == str(selected["sysroot"])


def test_distinct_compile_and_link_sysroots_explicitly_unsupported(selected, tmp_path):
    with selected["cache"].open("a") as out:
        out.write(f"CMAKE_SYSROOT_COMPILE:PATH={tmp_path}\n")
    result = invoke(selected)
    assert result.returncode != 0
    assert "Separate compile/link sysroots are unsupported" in result.stderr


def test_final_cannot_invent_sysroot_not_selected_by_cmake(selected):
    selected["cache"].write_text(selected["cache"].read_text().replace(f"CMAKE_SYSROOT:PATH={selected['sysroot']}\n", ""))
    result = invoke(selected, "--sysroot", selected["sysroot"])
    assert result.returncode != 0
    assert "not proven by selected CMake configuration" in result.stderr
    success(selected, "--sysroot", selected["sysroot"], final=False)


def test_competing_include_cannot_replace_selected_python_header_identity(selected, tmp_path):
    other = tmp_path / "other-python"
    other.mkdir()
    (other / "patchlevel.h").write_text("#define PY_MAJOR_VERSION 9\n#define PY_MINOR_VERSION 99\n")
    (other / "pyconfig.h").write_text("#define Py_DEBUG 1\n#define Py_GIL_DISABLED 1\n")
    with selected["cache"].open("a") as out:
        out.write(f"CMAKE_CXX_FLAGS:STRING=-I{other}\n")
    data = success(selected)
    assert data["compatibility"]["pyneat"]["abi_tag"] == "cp313"
    assert str(selected["include"] / "patchlevel.h") in data["compatibility"]["pyneat"]["preprocessor_input"]


def test_numpy_marker_requirements_preserved_without_policy_reinterpretation(selected):
    selected["repo"].joinpath("pyproject.toml").write_text('''[project]
requires-python = ">=3.9"
dependencies = [
  "numpy>=1.24; python_version < '3.13'",
  "numpy>=2.1; python_version >= '3.13'",
]
[project.optional-dependencies]
torch = ["torch>=2.3.0"]
''')
    data = success(selected)["compatibility"]["pyneat"]
    assert data["numpy_requirements"] == ["numpy>=1.24; python_version < '3.13'", "numpy>=2.1; python_version >= '3.13'"]
    assert all(requirement in data["numpy"] for requirement in data["numpy_requirements"])
    assert data["torch_requirements"] == ["torch>=2.3.0"]
    assert data["torch"].endswith(" (optional)")
    assert data["source_python_min"] == "3.9"


@pytest.mark.parametrize("project, expected", [
    ('[project]\ndependencies = ["numpy"]\n', ["numpy"]),
    ('[project]\ndependencies = []\n', None),
    (None, None),
])
def test_numpy_unversioned_or_unavailable_not_invented(selected, project, expected):
    path = selected["repo"] / "pyproject.toml"
    if project is None:
        path.unlink()
    else:
        path.write_text(project)
    data = success(selected)["compatibility"]["pyneat"]
    assert data.get("numpy_requirements") == expected
    if expected is None:
        assert "numpy" not in data


def test_invalid_pyproject_is_not_silently_reinterpreted(selected):
    (selected["repo"] / "pyproject.toml").write_text('[project\ndependencies = ["numpy"]')
    result = invoke(selected)
    assert result.returncode != 0
    assert "TOMLDecodeError" in result.stderr


@pytest.mark.parametrize("key", ["CMAKE_CXX_COMPILER_ARG1", "CMAKE_CXX_COMPILER_TARGET", "CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN"])
@pytest.mark.parametrize("location", ["cache", "cmake_compiler"])
def test_unhandled_compiler_invocation_context_fails_closed(selected, key, location):
    with selected[location].open("a") as out:
        out.write(f"{key}:STRING=additional-context\n" if location == "cache" else f'set({key} "additional-context")\n')
    result = invoke(selected)
    assert result.returncode != 0
    assert f"Unsupported compiler invocation context: {key}" in result.stderr


@pytest.mark.parametrize("profile", ["matching", "different", "link-only"])
def test_link_sysroot_override_is_not_silently_ignored(selected, tmp_path, profile):
    text = selected["cache"].read_text()
    if profile == "link-only":
        text = text.replace(f"CMAKE_SYSROOT:PATH={selected['sysroot']}\n", "")
    root = tmp_path if profile == "different" else selected["sysroot"]
    selected["cache"].write_text(text + f"CMAKE_SYSROOT_LINK:PATH={root}\n")
    if profile == "matching":
        success(selected)
    else:
        result = invoke(selected)
        assert result.returncode != 0
        assert ("Link-only sysroot is unsupported" if profile == "link-only" else
                "Separate compile/link sysroots are unsupported") in result.stderr
