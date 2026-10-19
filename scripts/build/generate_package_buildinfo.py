#!/usr/bin/env python3
"""Generate installed package provenance metadata for Neat builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

try:
    import tomllib
except ModuleNotFoundError:  # Match the wheel assembler's Python 3.9/3.10 fallback.
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        from pip._vendor import tomli as tomllib


# Policy / packaging-derived constants. These mirror the Debian `Depends:` floors
# and supported-host list; they change rarely and live here (one place, in code)
# so the docs never hardcode them — the Compatibility page renders from the
# emitted `compatibility` block. Keep these in sync with the packaging Depends
# (sima-neat / neat-runtime control files) and the elxr-sdk-host-setup docs.
COMPAT_DEFAULTS = {
    "target": {
        "board": "any Modalix DevKit",
        "arch": "aarch64",
        "glibc_min": "2.34",        # neat-runtime Depends libc6 (>= 2.34)
        "opencv": "4.10",           # sima-neat Depends libopencv-core410
        "gstreamer": "1.0",         # sima-neat Depends libgstreamer1.0-0
        "cxx_std": "20",
    },
    "host_sdk": {
        "os": [
            "Ubuntu 22.04 / 24.04",
            "Windows 11 (WSL2 + Docker Engine inside WSL)",
            "macOS (Colima + Docker)",
        ],
    },
    "pyneat": {},
    "tools": {"sima_cli": "latest"},
}


def git_value(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def split_ref(value: str) -> dict[str, str]:
    if not value:
        return {}
    if ":" in value:
        ref, spec = value.split(":", 1)
        return {"ref": ref, "spec": spec}
    return {"ref": value}


def parse_pyproject_floors(repo_root: Path) -> dict:
    """Preserve marker-qualified requirements, not a fictitious universal floor."""
    try:
        project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8")).get("project", {})
    except FileNotFoundError:
        return {}
    out = {}
    dependencies = project.get("dependencies", [])
    optional = [item for group in project.get("optional-dependencies", {}).values() for item in group]
    for name in ("numpy", "torch"):
        def matches(requirement: str) -> bool:
            match = re.match(r"[A-Za-z0-9_.-]+", requirement)
            return bool(match and match[0].lower().replace("_", "-") == name)
        required = [item for item in dependencies if matches(item)]
        requirements = required + [item for item in optional if matches(item) and item not in required]
        if requirements:
            out[name + "_requirements"] = requirements
            out[name] = " | ".join(requirements) + (" (optional)" if not required else "")
    requires_python = project.get("requires-python", "")
    if requires_python:
        out["requires_python"] = requires_python
        minimum = re.fullmatch(r"\s*>=\s*([0-9.]+)\s*", requires_python)
        if minimum:
            out["source_python_min"] = minimum[1]
    return out


def read_sdk_release(path: Path) -> dict[str, str]:
    """Parse /etc/sdk-release for `SDK Version` / `eLXr Version`. Empty if absent
    (e.g. building outside the SDK container)."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    def clean_version(raw: str) -> str:
        # "2.0.0_Palette_SDK_neat_main_780365a" -> "2.0.0" for display; fall back to raw.
        m = re.match(r"\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", raw)
        return m.group(1) if m else raw.strip()

    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key == "SDK Version":
            out["sdk"] = clean_version(val)
            out["sdk_release"] = val
        elif key == "eLXr Version":
            out["elxr"] = clean_version(val)
            out["elxr_release"] = val
    return out


def read_cmake_cache(path: Path, repo_root: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([^/#][^:]*):[^=]+=(.*)", line)
        if match:
            values[match[1]] = match[2]
    source = values.get("CMAKE_HOME_DIRECTORY", "")
    if not source or Path(source).resolve() != repo_root:
        raise ValueError("CMake cache does not belong to --repo-root")
    if "SIMANEAT_BUILD_PYTHON" not in values:
        raise ValueError("CMake cache lacks SIMANEAT_BUILD_PYTHON")
    return values


def selected_path(explicit: Path | None, cached: str, name: str) -> Path | None:
    if explicit and cached and explicit.resolve() != Path(cached).resolve():
        raise ValueError(f"{name} override contradicts selected CMake cache")
    path = explicit or (Path(cached) if cached else None)
    if path is not None and not path.exists():
        raise ValueError(f"{name} does not exist: {path}")
    return path.resolve() if path is not None else None


def cmake_compiler(cache_path: Path, cache: dict[str, str]) -> dict[str, str]:
    version = ".".join(cache.get(f"CMAKE_CACHE_{part}_VERSION", "")
                       for part in ("MAJOR", "MINOR", "PATCH"))
    path = cache_path.parent / "CMakeFiles" / version / "CMakeCXXCompiler.cmake"
    text = path.read_text(encoding="utf-8")
    for key in ("CMAKE_CXX_COMPILER_ARG1", "CMAKE_CXX_COMPILER_TARGET", "CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN"):
        match = re.search(r'set\(' + key + r' "([^"\n]*)"\)', text)
        if cache.get(key, "") or (match and match[1]):
            raise ValueError(f"Unsupported compiler invocation context: {key}")
    fields = {}
    for key in ("CMAKE_CXX_COMPILER", "CMAKE_CXX_COMPILER_ID", "CMAKE_CXX_COMPILER_VERSION"):
        match = re.search(r'set\(' + key + r' "([^"\n]+)"\)', text)
        if not match:
            raise ValueError(f"Missing {key} in {path}")
        fields[key] = match[1]
    cached = cache.get("CMAKE_CXX_COMPILER", "")
    selected_path(Path(fields["CMAKE_CXX_COMPILER"]), cached, "C++ compiler")
    return {"path": fields["CMAKE_CXX_COMPILER"],
            "id": fields["CMAKE_CXX_COMPILER_ID"],
            "version": fields["CMAKE_CXX_COMPILER_VERSION"],
            "cmake_metadata": str(path.resolve())}


def configured_cxx_flags(cache: dict) -> list[str]:
    flags = cache.get("CMAKE_CXX_FLAGS", "") + " " + cache.get(
        "CMAKE_CXX_FLAGS_" + cache.get("CMAKE_BUILD_TYPE", "").upper(), "")
    return shlex.split(flags)


def configured_sysroot(args: argparse.Namespace, cache: dict) -> Path | None:
    root = cache.get("CMAKE_SYSROOT", "")
    compile_root = cache.get("CMAKE_SYSROOT_COMPILE", "")
    if root and compile_root and Path(root).resolve() != Path(compile_root).resolve():
        raise ValueError("Separate compile/link sysroots are unsupported by package metadata")
    configured = compile_root or root
    flags = iter(configured_cxx_flags(cache))
    flag_roots = []
    for flag in flags:
        if flag in ("--sysroot", "-isysroot"):
            value = next(flags, "")
        elif flag.startswith("--sysroot="):
            value = flag.split("=", 1)[1]
        elif flag.startswith("-isysroot"):
            value = flag[len("-isysroot"):].removeprefix("=")
        else:
            continue
        if not value or not Path(value).is_absolute():
            raise ValueError("Configured compile sysroot flag must name an absolute path")
        flag_roots.append(value)
    for value in flag_roots:
        if configured and Path(configured).resolve() != Path(value).resolve():
            raise ValueError("C++ flags contradict configured compile sysroot")
        configured = value
    link_root = cache.get("CMAKE_SYSROOT_LINK", "")
    if link_root:
        if not configured:
            raise ValueError("Link-only sysroot is unsupported by package metadata")
        if Path(link_root).resolve() != Path(configured).resolve():
            raise ValueError("Separate compile/link sysroots are unsupported by package metadata")
    if args.phase == "final" and args.sysroot and not configured:
        raise ValueError("Final sysroot override is not proven by selected CMake configuration")
    return selected_path(args.sysroot, configured, "sysroot")


def python_headers(include: Path, compiler: dict, sysroot: Path | None, cache: dict) -> dict:
    # Use the selected host compiler's preprocessor to follow target multiarch
    # pyconfig.h dispatch. Never import/execute a target interpreter or extension.
    for name in ("Python.h", "patchlevel.h", "pyconfig.h"):
        if not (include / name).is_file():
            raise ValueError(f"Missing target Python header: {include / name}")
    # Match CMake's target-include precedence over ambient CXXFLAGS. Include the
    # selected files explicitly so a competing -I cannot attest different bytes.
    command = [compiler["path"], "-I", str(include), *configured_cxx_flags(cache),
               "-dM", "-E", "-x", "c++"]
    if sysroot:
        command += ["--sysroot=" + str(sysroot)]
    command += ["-"]
    source = "".join(f"#include {json.dumps(str(include / name))}\n"
                     for name in ("patchlevel.h", "pyconfig.h"))
    result = subprocess.run(command, input=source, text=True, capture_output=True, check=True)
    macros = dict(re.findall(r"^#define (\w+) ([^\n]+)$", result.stdout, re.MULTILINE))
    major = int(macros["PY_MAJOR_VERSION"])
    minor = int(macros["PY_MINOR_VERSION"])
    if major != 3 or minor < 8:
        raise ValueError("Target Python ABI detection supports CPython 3.8 or newer")
    flags = {flag for macro, flag in (("Py_DEBUG", "d"), ("Py_GIL_DISABLED", "t"))
             if macros.get(macro, "0") == "1"}
    for macro in ("Py_DEBUG", "Py_GIL_DISABLED"):
        if macros.get(macro, "0") not in ("0", "1"):
            raise ValueError(f"Unresolved Python ABI flag {macro}")
    return {"major": major, "minor": minor, "flags": flags,
            "include_dir": str(include),
            "header_sha256": {name: hashlib.sha256((include / name).read_bytes()).hexdigest()
                              for name in ("Python.h", "patchlevel.h", "pyconfig.h")},
            "preprocessor_command": command, "preprocessor_input": source,
            "preprocessor_macros_sha256": hashlib.sha256(result.stdout.encode()).hexdigest()}


def extension_abi(suffix: str) -> tuple[str, str, str]:
    match = re.search(r"\.cpython-(3[0-9]+)([a-z]*)-([A-Za-z0-9_-]+)\.so$", suffix)
    if not match or set(match[2]) - {"d", "t"} or len(set(match[2])) != len(match[2]):
        raise ValueError(f"Unsupported or missing CPython extension ABI suffix: {suffix}")
    return match[1], match[2], match[3]


def selected_build(args: argparse.Namespace, repo_root: Path) -> dict:
    cache = read_cmake_cache(args.cmake_cache, repo_root) if args.cmake_cache else {}
    if args.phase == "final" and not args.cmake_cache:
        raise ValueError("Final metadata requires --cmake-cache")
    enabled = args.python_enabled
    if cache:
        value = cache["SIMANEAT_BUILD_PYTHON"].upper()
        if value not in ("ON", "OFF", "TRUE", "FALSE", "1", "0"):
            raise ValueError("Invalid SIMANEAT_BUILD_PYTHON in CMake cache")
        cached_enabled = "on" if value in ("ON", "TRUE", "1") else "off"
        if enabled != "unknown" and enabled != cached_enabled:
            raise ValueError("Python profile contradicts selected CMake cache")
        enabled = cached_enabled
    sysroot = configured_sysroot(args, cache)
    compiler = cmake_compiler(args.cmake_cache, cache) if cache else {}
    target_sdk = {"status": "unavailable"}
    if sysroot:
        target_sdk["sysroot"] = str(sysroot)
    selected_release = sysroot / "etc/sdk-release" if sysroot else None
    if args.target_sdk_release_file:
        if not args.target_sdk_release_file.is_file():
            raise ValueError("Target SDK release override does not exist")
        if selected_release and selected_release.is_file():
            if read_sdk_release(selected_release) != read_sdk_release(args.target_sdk_release_file):
                raise ValueError("Target SDK release override contradicts selected sysroot")
        selected_release = args.target_sdk_release_file
    if selected_release and selected_release.is_file():
        release = read_sdk_release(selected_release)
        target_sdk.update(release)
        target_sdk["release_file"] = str(selected_release.resolve())
        target_sdk["release_sha256"] = hashlib.sha256(selected_release.read_bytes()).hexdigest()
        target_sdk["status"] = "observed" if release else "unavailable"
    if compiler:
        target_sdk["compiler"] = compiler
        target_sdk["toolchain"] = f'{compiler["id"]} {compiler["version"]} ({compiler["path"]})'
    if args.toolchain:
        # Retain the caller's label, separately from observed compiler identity.
        target_sdk["toolchain_override"] = args.toolchain
    pyneat = {"status": "disabled" if enabled == "off" else "unavailable"}
    if enabled == "off":
        if args.python_extension:
            raise ValueError("Python extension supplied for Python-disabled build")
    else:
        includes = cache.get("Python_INCLUDE_DIR", "") or cache.get("_Python_INCLUDE_DIR", "")
        include = selected_path(args.python_include_dir, includes, "Python include directory")
        if include:
            for header in ("Python.h", "patchlevel.h", "pyconfig.h"):
                if not (include / header).is_file():
                    raise ValueError(f"Missing target Python header: {include / header}")
        if args.phase == "final":
            if enabled != "on" or not include or not args.python_extension:
                raise ValueError("Final Python-enabled metadata requires selected headers and built extension")
            extension = args.python_extension.resolve()
            if not extension.is_file() or extension.stat().st_size == 0:
                raise ValueError("Selected Python extension is missing or empty")
            if not extension.is_relative_to(args.cmake_cache.parent.resolve()):
                raise ValueError("Python extension is outside selected CMake build")
            info = python_headers(include, compiler, sysroot, cache)
            tag, flags, platform = extension_abi(extension.name)
            if tag != f'{info["major"]}{info["minor"]}' or set(flags) != info.pop("flags"):
                raise ValueError("Built extension ABI contradicts target Python headers")
            suffix = cache.get("PYNEAT_EXT_SUFFIX", "")
            if suffix and extension_abi(suffix) != (tag, flags, platform):
                raise ValueError("Built extension ABI contradicts selected CMake suffix")
            pyneat.update(info)
            pyneat.update(status="observed", python_abi=f'CPython {info["major"]}.{info["minor"]} (cp{tag}{flags})',
                          abi_tag=f"cp{tag}{flags}", extension_platform=platform,
                          extension=str(extension), extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest())
    return {"target_sdk": target_sdk, "pyneat": pyneat}


def build_compatibility(args: argparse.Namespace, repo_root: Path, selected: dict) -> dict:
    """Policy floors are not observed build-host or selected-target identities."""
    import copy
    compat = copy.deepcopy(COMPAT_DEFAULTS)
    compat["target"]["platform_sw"] = args.platform_version.split("+", 1)[0]
    compat["pyneat"].update(parse_pyproject_floors(repo_root))
    compat["pyneat"].update(selected["pyneat"])
    host = read_sdk_release(args.sdk_release_file)
    compat["host_sdk"].update(host)
    compat["host_sdk"]["status"] = "observed" if host else "unavailable"
    if host:
        compat["host_sdk"]["release_file"] = str(args.sdk_release_file.resolve())
        compat["host_sdk"]["release_sha256"] = hashlib.sha256(args.sdk_release_file.read_bytes()).hexdigest()
    compat["target_sdk"] = selected["target_sdk"]
    return compat


def source_state(repo_root: Path, output_path: Path) -> dict:
    sha = git_value(repo_root, "rev-parse", "HEAD")
    if not sha:
        return {"git_sha": "", "git_dirty": None, "source_status": "unavailable"}
    # Record actual checkout state, not GITHUB_SHA masquerading as built bytes.
    result = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"],
                            cwd=repo_root, text=True, capture_output=True, check=True)
    # Do not hash this generator's own output if a caller stages it in the
    # source tree. Other nonignored untracked bytes remain part of the record.
    excluded = []
    output = output_path.absolute()
    if output.is_relative_to(repo_root):
        excluded = [output.relative_to(repo_root).as_posix()]
    paths = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repo_root)
    untracked = {}
    for raw in sorted(filter(None, paths.split(b"\0"))):
        name = os.fsdecode(raw)
        if name in excluded:
            continue
        path = repo_root / name
        if path.is_symlink():
            untracked[name] = {"type": "symlink", "target": os.readlink(path)}
        else:
            untracked[name] = {"type": "file", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "mode": path.stat().st_mode & 0o7777}
    untracked_digest = hashlib.sha256(json.dumps(untracked, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"], cwd=repo_root)
    return {"git_sha": sha, "git_checkout_ref": git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(result.stdout), "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "git_status_porcelain": result.stdout.splitlines(),
            "untracked_files": untracked, "untracked_files_sha256": untracked_digest,
            "untracked_identity_exclusions": excluded,
            "source_status": "dirty" if result.stdout else "clean"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-name", required=True)
    parser.add_argument("--package-version", required=True)
    parser.add_argument("--platform-version", required=True)
    parser.add_argument("--vulcan-env", default="dev")
    parser.add_argument("--scheduler-ref", default="")
    parser.add_argument("--llima-ref", default="")
    parser.add_argument("--sdk-release-file", type=Path, default=Path("/etc/sdk-release"),
                        help="Observed build-host SDK release file; never used as selected target SDK identity.")
    parser.add_argument("--toolchain", default="",
                        help="Cross toolchain version string (e.g. 'aarch64-linux-gnu-g++ 12.2.0').")
    parser.add_argument("--phase", choices=("provisional", "final"), default="provisional")
    parser.add_argument("--cmake-cache", type=Path)
    parser.add_argument("--sysroot", type=Path)
    parser.add_argument("--target-sdk-release-file", type=Path)
    parser.add_argument("--python-enabled", choices=("on", "off", "unknown"), default="unknown")
    parser.add_argument("--python-include-dir", type=Path)
    parser.add_argument("--python-extension", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_path = args.output

    branch = (
        os.environ.get("GITHUB_HEAD_REF")
        or os.environ.get("GITHUB_REF_NAME")
        or git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    )
    if branch == "HEAD":
        branch = os.environ.get("GITHUB_REF_NAME", "")

    state = source_state(repo_root, output_path)
    full_sha = state["git_sha"]
    selected = selected_build(args, repo_root)
    short_sha = full_sha[:12] if full_sha else ""
    ref_type = os.environ.get("GITHUB_REF_TYPE") or (
        "tag" if git_value(repo_root, "describe", "--tags", "--exact-match") else "branch"
    )
    repository = os.environ.get("GITHUB_REPOSITORY", "sima-neat/core")
    repo_key = repository.split("/", 1)[-1]
    branch_key = quote(branch, safe="") if branch else ""

    vulcan = {
        "environment": args.vulcan_env.strip() or "dev",
        "repository": repo_key,
        "ref": branch,
        "ref_key": branch_key,
        "ref_type": ref_type,
        "spec": short_sha,
    }

    payload = {
        "_note": "Generated by scripts/build/generate_package_buildinfo.py; do not hand-edit.",
        "format_version": 1,
        "package": args.package_name,
        "version": args.package_version,
        "source": "local" if state["git_dirty"] is not False or not os.environ.get("GITHUB_RUN_ID") else "vulcan",
        "metadata_phase": args.phase,
        "vulcan": vulcan,
        "build": {
            "repository": repository,
            "git_ref": branch,
            "git_ref_type": ref_type,
            **state,
            "ci_requested_git_sha": os.environ.get("GITHUB_SHA", ""),
            "git_short_sha": short_sha,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        },
        "dependencies": {},
        "compatibility": build_compatibility(args, repo_root, selected),
    }
    if args.scheduler_ref:
        payload["dependencies"]["scheduler"] = split_ref(args.scheduler_ref)
    if args.llima_ref:
        payload["dependencies"]["llima"] = split_ref(args.llima_ref)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
