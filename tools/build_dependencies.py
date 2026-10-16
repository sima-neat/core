#!/usr/bin/env python3
"""Prepare Core's native dependencies without network or device side effects.

Source and prebuilt packages meet at the same installed-package boundary. This
module intentionally knows only the fixed Kernels -> Runner -> Scheduler order, not a second
build graph or a workspace manifest.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import posixpath
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

COMPONENTS = ("kernels", "runner", "scheduler")
PACKAGE_NAMES = {"kernels": "SimaKernels", "runner": "SimaRunner",
                 "scheduler": "SimaScheduler", "runtime_compat": "SimaRuntime"}
DEBIAN_ROLES = {"kernels": "simaai-kernels-dev", "runner": "simaai-runner",
                "scheduler": "simaai-scheduler", "runtime_compat": "simaai-runtime"}
PROJECT_FILES = {
    "kernels": "include/simaai/kernels/tensor.h",
    "runner": "cmake/SimaRunnerConfig.cmake.in",
    "scheduler": "cmake/SimaSchedulerConfig.cmake.in",
}


def source_is_valid(component: str, directory: Path) -> bool:
    return (directory / "CMakeLists.txt").is_file() and (directory / PROJECT_FILES[component]).is_file()


def select(core: Path, env: dict[str, str]) -> dict[str, dict[str, str]]:
    selected = {}
    for component in COMPONENTS:
        key = "NEAT_" + component.upper()
        source, package = env.get(key + "_SOURCE", ""), env.get(key + "_PACKAGE_DIR", "")
        if source and package:
            raise ValueError(f"Set only one of {key}_SOURCE and {key}_PACKAGE_DIR")
        if source:
            path = Path(source).resolve()
            if not source_is_valid(component, path):
                raise ValueError(f"{key}_SOURCE is not a {component} source root: {path}")
            kind = "source"
        elif package:
            path, kind = Path(package).resolve(), "package"
            if not path.is_dir():
                raise ValueError(f"{key}_PACKAGE_DIR does not exist: {path}")
        else:
            sibling = core.parent / component
            if source_is_valid(component, sibling):
                path, kind = sibling.resolve(), "source"
            elif (sibling / "dist/package/usr").is_dir():
                path, kind = (sibling / "dist/package").resolve(), "package"
            else:
                continue
        selected[component] = {"kind": kind, "path": str(path)}
    return selected


def package_prefix(root: Path) -> Path:
    return root / "usr" if (root / "usr").is_dir() else root


def find_config(root: Path, name: str) -> Path:
    matches = list(package_prefix(root).glob(f"lib*/**/{name}Config.cmake"))
    if len(matches) != 1:
        raise ValueError(f"Expected one installed {name}Config.cmake under {root}; found {len(matches)}")
    return matches[0].parent


def receipt(root: Path, name: str) -> dict:
    path = package_prefix(root) / "share" / name / "buildinfo.json"
    if not path.is_file():
        raise ValueError(f"Missing dependency build receipt: {path}. Rebuild and install the selected CMake package.")
    data = json.loads(path.read_text())
    expected = {"SimaKernels": "kernels", "SimaRunner": "runner",
                "SimaScheduler": "scheduler", "SimaRuntime": "runtime-compat"}[name]
    if data.get("schema_version") not in ((1, 2) if name == "SimaKernels" else (1,)) or data.get("component") != expected:
        raise ValueError(f"Unsupported {name} build receipt: {path}")
    if name != "SimaKernels":
        verify_runtime_payload(package_prefix(root), data, name)
    if name == "SimaKernels" and data.get("schema_version") == 2:
        verify_kernel_payload(package_prefix(root), data)
    return data


def verify_runtime_payload(prefix: Path, info: dict, name: str = "SimaRunner") -> None:
    exclusions = (f'share/{name}/buildinfo.json',)
    if name == "SimaScheduler":
        exclusions += (f'share/{name}/platform-install-manifest.json',)
    if info.get('payload_exclusions') != list(exclusions):
        raise ValueError('Unsupported Runtime payload exclusions')
    actual = {}
    for path in sorted(prefix.rglob('*')):
        name = path.relative_to(prefix).as_posix()
        if name in exclusions:
            continue
        if path.is_symlink():
            target = os.readlink(path)
            if os.path.isabs(target) or not path.resolve().is_relative_to(prefix):
                raise ValueError('Non-relocatable Runtime payload link')
            actual[name] = {'kind': 'link', 'target': target}
        elif path.is_file():
            actual[name] = {'kind': 'file', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'mode': path.stat().st_mode & 0o7777}
    digest = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if not info.get('installed_files') or actual != info['installed_files'] or digest != info.get('installed_payload_sha256'):
        raise ValueError('Installed Runtime payload differs from package receipt')
    identity = {key: value for key, value in info.items() if key != 'package_sha256'}
    if hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != info.get('package_sha256'):
        raise ValueError('Runtime package identity mismatch')


def verify_kernel_payload(prefix: Path, info: dict) -> None:
    expected = {name: item for name, item in info.get("installed_files", {}).items()
                if not name == "share/sima-kernels/tests/package.json"}
    actual = {}
    for path in sorted(prefix.rglob("*")):
        name = path.relative_to(prefix).as_posix()
        if name == "share/SimaKernels/buildinfo.json" or name == "share/sima-kernels/tests/package.json" or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            actual[name] = {"kind": "link", "target": os.readlink(path)}
        elif path.is_file():
            actual[name] = {"kind": "file", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mode": path.stat().st_mode & 0o777}
    if not expected or actual != expected:
        raise ValueError("Kernels installed payload differs from its build metadata")
    digest = hashlib.sha256(json.dumps(info["installed_files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != info.get("installed_payload_sha256"):
        raise ValueError("Kernels installed payload inventory digest mismatch")


def build_kernels(source: Path, destination: Path, build: Path, jobs: int,
                  toolchain: str | None, clean: bool) -> None:
    """Standard CMake SDK build; DEB is a projection of this exact install."""
    for directory in (build, destination):
        if directory == source or source.is_relative_to(directory) or directory.is_relative_to(source):
            raise ValueError("Kernels build/staging must be outside the source tree")
    build.parent.mkdir(parents=True, exist_ok=True)
    lock = (build.parent / ("." + build.name + ".kernels-build.lock")).open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = build / "CMakeCache.txt"
        if build.exists() and any(build.iterdir()):
            if not cache.is_file() or f"CMAKE_HOME_DIRECTORY:INTERNAL={source}" not in cache.read_text().splitlines():
                raise ValueError("Refusing an unowned Kernels build directory")
        if clean and build.exists():
            shutil.rmtree(build)
        environment = {key: value for key, value in os.environ.items() if key not in
                       ("CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CMAKE_TOOLCHAIN_FILE")}
        selected_toolchain = Path(toolchain).resolve() if toolchain else source / "cmake/toolchains/a65-linux.cmake"
        command = ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
                   "-DSIMA_KERNELS_PROFILE=a65", "-DBUILD_TESTING=OFF", "-DCMAKE_BUILD_TYPE=Release",
                   "-DCMAKE_INSTALL_PREFIX=/usr", "-DCMAKE_INSTALL_LIBDIR=lib/aarch64-linux-gnu",
                   f"-DCMAKE_TOOLCHAIN_FILE={selected_toolchain}",
                   "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF", "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF"]
        subprocess.run(command, cwd=source, check=True, env=environment)
        subprocess.run(["cmake", "--build", str(build), "--parallel", str(jobs)], cwd=source, check=True, env=environment)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".kernels-stage-", dir=destination.parent) as temporary:
            stage = Path(temporary)
            subprocess.run(["cmake", "--install", str(build)], cwd=source, check=True,
                           env={**environment, "DESTDIR": str(stage)})
            info = receipt(stage, "SimaKernels")
            name = f"simaai-kernels-dev_0.1.0+local.{info.get('package_sha256', info['source_sha256'])[:12]}_arm64.deb"
            subprocess.run([sys.executable, str(stage / "usr/share/SimaKernels/tools/package_metadata.py"),
                            "deb", "--prefix", str(stage / "usr"), "--output", str(stage / name)],
                           cwd=source, check=True, env=environment)
            previous = stage.parent / (stage.name + "-previous")
            if destination.exists():
                receipt(destination, "SimaKernels")
                destination.rename(previous)
            try:
                stage.rename(destination)
            except OSError:
                if previous.exists():
                    previous.rename(destination)
                raise
            if previous.exists():
                shutil.rmtree(previous)
    finally:
        lock.close()



def build_runtime_source(component, path, destination, output, packages, jobs, clean):
    command = ["bash", str(path / "build.sh"), "--all", "--package-dir", str(destination),
               "--build-dir", str(output / "build" / component), "--jobs", str(jobs),
               "--kernels-package", packages["kernels"]["package"]]
    if clean:
        command += ["--clean"]
    toolchain = os.environ.get("CMAKE_TOOLCHAIN_FILE") or packages["kernels"]["buildinfo"].get("toolchain", {}).get("path", "")
    if toolchain:
        command += ["--toolchain", str(Path(toolchain).resolve())]
    child_env = dict(os.environ)
    for key in ("NEAT_KERNELS_SOURCE", "NEAT_KERNELS_PACKAGE_DIR", "NEAT_KERNELS_PACKAGE", "SimaKernels_DIR"):
        child_env.pop(key, None)
    if component == "scheduler":
        command += ["--runner-package", packages["runner"]["package"]]
    codecs = child_env.get("NEAT_CODECS_PACKAGE_DIR")
    if component == "scheduler" and codecs:
        command += ["--codecs-package", str(Path(codecs).resolve())]
    if component == "runner":
        for archive in platform_runtime_archives(child_env):
            command += ["--platform-runtime-deb", str(archive)]
    extra = shlex.split(child_env.get("NEAT_" + component.upper() + "_CMAKE_ARGS", ""))
    if extra:
        command += ["--", *extra]
    subprocess.run(command, cwd=path, check=True, env=child_env)
    return destination


def codec_prefix_inventory(prefix: Path) -> dict:
    result = {}
    for path in sorted(prefix.rglob('*')):
        name = 'usr/' + path.relative_to(prefix).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            if os.path.isabs(target) or not path.resolve().is_relative_to(prefix):
                raise ValueError(f'Non-relocatable Codecs payload link: {name}')
            result[name] = {'kind': 'symlink', 'target': target}
        elif path.is_file():
            result[name] = {'kind': 'file', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'mode': path.stat().st_mode & 0o7777}
    return result


@contextmanager
def dependency_archive(path: Path):
    """Drain trailing tar padding; close and reap a rejected archive producer."""
    process = subprocess.Popen(['dpkg-deb', '--fsys-tarfile', str(path)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            yield archive
        for _ in iter(lambda: process.stdout.read(1024 * 1024), b''):
            pass
        if process.wait() != 0:
            raise ValueError(f'Cannot inspect dependency archive: {path}')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def dependency_archive_path(member):
    name = member.name.removeprefix('./')
    if name in ('', '.') and member.isdir():
        return ''
    normalized = PurePosixPath(name)
    if normalized.is_absolute() or str(normalized) != name or any(p in ('..', '.') for p in normalized.parts):
        raise ValueError(f'Unsafe dependency archive path: {member.name}')
    return name


def codec_deb_inventory(path: Path) -> dict:
    result = {};seen = set()
    with dependency_archive(path) as archive:
        for member in archive:
            name = dependency_archive_path(member)
            if name in seen:
                raise ValueError(f'Duplicate Codecs archive entry: {name}')
            seen.add(name)
            if name == '':
                continue
            if name != 'usr' and not name.startswith('usr/'):
                raise ValueError(f'Codecs archive payload outside usr: {name}')
            if member.isfile():
                result[name] = {'kind': 'file', 'sha256': hashlib.sha256(archive.extractfile(member).read()).hexdigest(),
                                'mode': member.mode & 0o7777}
            elif member.issym():
                result[name] = {'kind': 'symlink', 'target': member.linkname}
            elif not member.isdir():
                raise ValueError(f'Unsupported Codecs archive entry: {name}')
    return result


def compose_external_codecs(runner: dict, destination: Path | None) -> dict | None:
    """Audit an external installed provider; never add a native build step."""
    consumed = runner.get('external_dependencies', {}).get('SimaCodecs')
    configured = runner.get('build_configuration', {})
    flag = configured.get('SIMA_RUNTIME_CODECS')
    if flag is not None:
        enabled = flag.upper() in ('ON', 'TRUE', '1', 'YES')
        if enabled and not consumed:
            raise ValueError('Enabled Runtime Codec wrappers lack a consumed SimaCodecs receipt')
        if not enabled:
            return None
    if not consumed:
        return None
    selected = os.environ.get('NEAT_CODECS_PACKAGE_DIR')
    if not selected:
        raise ValueError('Runtime Codec wrappers require NEAT_CODECS_PACKAGE_DIR with the exact consumed installed prefix and runtime/dev DEBs; no artifact fallback')
    root = Path(selected).resolve(strict=True);prefix = package_prefix(root)
    config = find_config(root, 'SimaCodecs')
    path = prefix / 'share/SimaCodecs/buildinfo.json'
    info = json.loads(path.read_text())
    if info.get('component') != 'codecs' or info.get('schema_version') != 1 or info.get('architecture') != 'aarch64':
        raise ValueError('Unsupported selected Codecs receipt/architecture')
    inventory = codec_prefix_inventory(prefix)
    digest = hashlib.sha256(''.join(f"{p.relative_to(prefix).as_posix()}:{hashlib.sha256(p.read_bytes()).hexdigest()}\n"
                                for p in sorted(prefix.rglob('*')) if p.is_file()).encode()).hexdigest()
    if consumed.get('package_sha256') != digest or consumed.get('config_sha256') != hashlib.sha256((config / 'SimaCodecsConfig.cmake').read_bytes()).hexdigest():
        raise ValueError('Selected Codecs installed prefix differs from the consumed Runtime Codec payload')
    for name, recorded in info.get('files', {}).items():
        actual = inventory.get(name)
        if not actual or any(actual.get(key) != value for key, value in recorded.items()):
            raise ValueError(f'Selected Codecs receipt/payload mismatch: {name}')
    sonames = set(info.get('sonames', {}).values())
    if not sonames or any(not any(Path(name).name == soname and item['kind'] == 'file' for name, item in inventory.items()) for soname in sonames):
        raise ValueError('Selected Codecs receipt lacks required runtime libraries')
    controls = []
    merged = {}
    roles = {}
    archive_root = root if (root / 'usr').is_dir() else prefix.parent
    for archive in sorted(archive_root.glob('*.deb')):
        files = codec_deb_inventory(archive)
        is_runtime = all(any(Path(name).name == soname and item['kind'] == 'file' for name, item in files.items()) for soname in sonames)
        is_dev = 'usr/' + (config / 'SimaCodecsConfig.cmake').relative_to(prefix).as_posix() in files
        if is_runtime == is_dev: # Neither, or a combined archive instead of the advertised pair.
            raise ValueError(f'Unexpected Codecs archive ownership role: {archive}')
        role = 'runtime' if is_runtime else 'dev'
        if role in roles:
            raise ValueError(f'Ambiguous Codecs {role} archive')
        fields = subprocess.check_output(['dpkg-deb', '-f', str(archive), 'Package', 'Version', 'Architecture', 'Depends'], text=True)
        control = dict(line.split(': ', 1) for line in fields.splitlines())
        subprocess.run(['dpkg', '--validate-version', control['Version']], check=True, capture_output=True)
        canonical = 'neat-codecs' if role == 'runtime' else 'neat-codecs-dev'
        if control.get('Package') != canonical:
            raise ValueError(f'Codecs archive has noncanonical {role} Package role')
        if control['Version'] != info.get('version') or control['Architecture'] != 'arm64':
            raise ValueError('Codecs archive control version/architecture differs from installed receipt')
        for name, item in files.items():
            if name in merged:
                raise ValueError(f'Codecs archives overlap ownership: {name}')
            merged[name] = item
        entry = {'role': role, 'path': str(archive), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), **control}
        roles[role] = entry;controls.append(entry)
    if set(roles) != {'runtime', 'dev'} or merged != inventory:
        raise ValueError('Codecs runtime/dev archives do not exactly cover the selected installed prefix')
    required = f"{roles['runtime']['Package']} (= {roles['runtime']['Version']})"
    if required not in [part.strip() for part in roles['dev'].get('Depends', '').split(',')]:
        raise ValueError('Codecs SDK archive must depend exactly on its selected runtime package/version')
    recorded = [entry for entry in runner.get('debian_dependencies', []) if entry.get('Package') == roles['runtime']['Package']]
    expected = roles['runtime']
    if len(recorded) != 1 or any(recorded[0].get(key) != expected[key] for key in ('Package', 'Version', 'Architecture')) or recorded[0].get('deb_sha256') != expected['sha256']:
        raise ValueError('Selected Codecs runtime archive differs from Runtime dependency control receipt')
    if codec_prefix_inventory(prefix) != inventory:
        raise ValueError('Codecs prefix changed during composition')
    if destination is not None:
        for entry in controls:
            archive = Path(entry['path'])
            if hashlib.sha256(archive.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Codecs archive changed during composition')
            target = destination / archive.name
            if target.exists():
                raise ValueError(f'Duplicate external dependency archive: {target}')
            shutil.copy2(archive, target)
    return {'kind': 'external-installed-package', 'package': str(root), 'prefix': str(prefix),
            'config': str(config), 'buildinfo': info, 'consumed_payload_sha256': digest, 'debian_packages': controls}


def platform_runtime_archives(environment: dict) -> list[Path]:
    """An explicit JSON list preserves spaces and never selects an SDK fallback."""
    value = environment.get('NEAT_RUNNER_PLATFORM_RUNTIME_DEBS', '[]')
    try:
        selected = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError('NEAT_RUNNER_PLATFORM_RUNTIME_DEBS must be a JSON array of archive paths') from error
    if not isinstance(selected, list) or any(not isinstance(p, str) or not p.strip() for p in selected):
        raise ValueError('NEAT_RUNNER_PLATFORM_RUNTIME_DEBS must be a JSON array of archive paths')
    result = [Path(p).resolve(strict=True) for p in selected]
    if len(result) != len(set(result)) or any(not p.is_file() for p in result):
        raise ValueError('Platform runtime archive paths must be distinct regular files')
    return result


def platform_archive_inventory(path: Path) -> dict:
    """Audit producer-owned runtime payloads without extracting or executing them."""
    result, seen = {}, set()
    with dependency_archive(path) as archive:
        for member in archive:
            name = dependency_archive_path(member)
            if name in seen:
                raise ValueError(f'Duplicate platform runtime archive entry: {name}')
            seen.add(name)
            if not name:
                continue
            if PurePosixPath(name).parts[0] not in ('usr', 'lib', 'opt'):
                raise ValueError(f'Unsupported platform runtime payload path: {name}')
            if member.isfile():
                result[name] = {'kind': 'file', 'sha256': hashlib.sha256(archive.extractfile(member).read()).hexdigest(),
                                'mode': member.mode & 0o7777}
            elif member.issym():
                target = member.linkname
                resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
                if not target or target.startswith('/') or PurePosixPath(resolved).parts[0] not in ('usr', 'lib', 'opt'):
                    raise ValueError(f'Non-relocatable platform runtime link: {name}')
                result[name] = {'kind': 'symlink', 'target': target}
            elif not member.isdir():
                raise ValueError(f'Unsupported platform runtime archive entry: {name}')
    for name in seen:
        if any(result.get(str(parent), {}).get('kind') == 'symlink' for parent in PurePosixPath(name).parents):
            raise ValueError(f'Platform runtime archive writes through a symlink: {name}')
    return result


def compose_external_runtimes(runner: dict, destination: Path | None) -> dict:
    """Compose only the exact backend runtime packages admitted by Runner.

    Runner owns backend selection, checked API admission and loader-to-archive
    identity checks. Core forwards and deploys those unchanged producer archives;
    it never embeds their libraries or infers backend support from a build number.
    """
    consumed = {name: entry for name, entry in runner.get('external_dependencies', {}).items()
                if isinstance(entry, dict) and 'debian_provider' in entry and 'loader_name' in entry}
    selected = platform_runtime_archives(os.environ)
    if not consumed:
        if selected:
            raise ValueError('Platform runtime archives selected but Runner records no backend providers')
        return {}
    controls = {}
    for archive in selected:
        fields = subprocess.check_output(['dpkg-deb', '-f', str(archive), 'Package', 'Version', 'Architecture'], text=True)
        control = dict(line.split(': ', 1) for line in fields.splitlines())
        if control.get('Architecture') not in ('arm64', 'all'):
            raise ValueError(f'Platform runtime archive architecture mismatch: {archive}')
        subprocess.run(['dpkg', '--validate-version', control['Version']], check=True, capture_output=True)
        if control['Package'] in controls:
            raise ValueError(f'Duplicate platform runtime Package: {control["Package"]}')
        controls[control['Package']] = (archive, {**control, 'deb_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()})
    result, used = {}, set()
    for name, provider in consumed.items():
        expected = provider['debian_provider']
        role = expected['Package']
        match = controls.get(role)
        if match is None or match[1] != expected:
            raise ValueError(f'{name}: missing exact Runner-consumed platform runtime DEB')
        recorded = [c for c in runner.get('debian_dependencies', []) if c.get('Package') == role]
        if len(recorded) != 1 or recorded[0] != expected:
            raise ValueError(f'{name}: Runner lacks its exact platform runtime dependency')
        archive, control = match
        used.add(role)
        files = platform_archive_inventory(archive)
        result[name] = {'kind': 'external-platform-runtime', 'archive': str(archive),
                        'debian_provider': control, 'installed_files': files}
    if used != set(controls):
        raise ValueError('Unconsumed platform runtime archive in selected cohort')
    if destination is not None:
        for role in sorted(used):
            archive, control = controls[role]
            if hashlib.sha256(archive.read_bytes()).hexdigest() != control['deb_sha256']:
                raise ValueError('Platform runtime archive changed during composition')
            target = destination / archive.name
            if target.exists():
                raise ValueError(f'Duplicate external runtime archive: {target}')
            shutil.copy2(archive, target)
    return result


def compose_external_boxrender(scheduler: dict, prefix: Path, destination: Path | None,
                              kernels: dict | None = None, runner: dict | None = None) -> dict | None:
    """Validate canonical rendering; retain composition for older installed Schedulers."""
    if scheduler.get('renderer') is not None:
        renderer = scheduler['renderer']
        expected = {'schema_version': 1, 'provider': 'kernels', 'operation': 'render',
                    'request_api_revision': 2, 'coordinates': ['int32', 'float32']}
        for name, selected in (('kernels', kernels), ('runner', runner)):
            digest = (selected or {}).get('contract_sha256', '')
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError(f'Canonical renderer requires selected {name} identity')
            expected[name + '_contract_sha256'] = digest
        if renderer != expected:
            raise ValueError('Canonical renderer differs from selected Kernels/Runner cohort')
        configuration = scheduler.get('build_configuration', {})
        if any(str(configuration.get(key, '')).upper() not in ('ON', 'TRUE', '1', 'YES')
               for key in ('SIMA_RUNTIME_RENDER', 'SIMA_SCHEDULER_RENDER_INT32_CAPABLE')):
            raise ValueError('Canonical renderer lacks checked int32 capability')
        if scheduler.get('external_dependencies', {}).get('BoxRender') or any(
                item.get('Package') in ('a65apps', 'a65apps-dev')
                for item in scheduler.get('debian_dependencies', [])):
            raise ValueError('Canonical renderer must not depend on retired a65apps')
        # Kernels and Runner are already composed by the normal dependency path.
        return None
    consumed = scheduler.get('external_dependencies', {}).get('BoxRender')
    required = scheduler.get('build_configuration', {}).get('SIMA_SCHEDULER_REQUIRE_RENDER_INT32', '').upper() in ('ON', 'TRUE', '1', 'YES')
    if not consumed:
        if required:
            raise ValueError('Required int32 renderer lacks its consumed BoxRender provider receipt')
        return None
    selected = os.environ.get('NEAT_BOXRENDER_PACKAGE_DIR')
    if not selected:
        raise ValueError('Selected Scheduler requires NEAT_BOXRENDER_PACKAGE_DIR with its exact a65apps/a65apps-dev provider pair')
    root = Path(selected).resolve(strict=True)
    if root.name == 'usr':
        root = root.parent
    verifier = prefix / 'share/SimaScheduler/tools/boxrender_package.py'
    if not verifier.is_file():
        raise ValueError('Selected Scheduler lacks its installed BoxRender package verifier')
    current = json.loads(subprocess.check_output([sys.executable, str(verifier), '--package', str(root)], text=True))
    if current != consumed:
        raise ValueError('Selected BoxRender provider differs from the Scheduler build receipt')
    runtime = current['packages']['runtime']
    recorded = [item for item in scheduler.get('debian_dependencies', []) if item.get('Package') == runtime['Package']]
    if len(recorded) != 1 or any(recorded[0].get(k) != runtime.get(k) for k in ('Package', 'Version', 'Architecture', 'deb_sha256')):
        raise ValueError('Scheduler lacks its exact selected BoxRender runtime dependency')
    if destination is not None:
        for control in current['packages'].values():
            archive = root / control['archive']
            if hashlib.sha256(archive.read_bytes()).hexdigest() != control['deb_sha256']:
                raise ValueError('BoxRender archive changed during composition')
            target = destination / archive.name
            if target.exists():
                raise ValueError(f'Duplicate external provider archive: {target}')
            shutil.copy2(archive, target)
    return {'kind': 'external-platform-package', 'package': str(root), 'buildinfo': current}


def prepare(core: Path, output: Path, selected: dict, jobs: int, headers_only: bool, clean: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    for component in COMPONENTS:
        if component not in selected:
            raise ValueError(f"No local {component} source/package. Set NEAT_{component.upper()}_SOURCE or NEAT_{component.upper()}_PACKAGE_DIR. TODO: remote artifact pulling is not implemented.")
    packages = {}
    for component in COMPONENTS:
        selection = selected.get(component)
        path = Path(selection["path"])
        if selection["kind"] == "source":
            if headers_only:
                raise ValueError(f"Header-only preparation will not compile {component}. Set NEAT_{component.upper()}_PACKAGE_DIR to an installed local package. TODO: remote artifact pulling.")
            destination = output / component
            if component == "kernels":
                build_kernels(path, destination, output / "build/kernels", jobs,
                              os.environ.get("CMAKE_TOOLCHAIN_FILE"), clean)
                path = destination
            else:
                path = build_runtime_source(component, path, destination, output, packages, jobs, clean)
        name = PACKAGE_NAMES[component]
        config = find_config(path, name)
        info = receipt(path, name)
        if not headers_only and info.get("architecture") != "aarch64":
            raise ValueError(f"Core requires an A65/aarch64 {component} package, not {info.get('architecture')}")
        packages[component] = {**selection, "package": str(path), "prefix": str(package_prefix(path)), "config": str(config), "buildinfo": info}
    # Dependency bytes must remain the selected cohort throughout construction.
    for component, item in packages.items():
        name = PACKAGE_NAMES[component]
        if receipt(Path(item["package"]), name) != item["buildinfo"]:
            raise ValueError(f"Selected {component} package changed during preparation")
    validate_cohort(packages)
    external_runtimes = compose_external_runtimes(packages['runner']['buildinfo'], None)
    external_codecs = compose_external_codecs(packages['scheduler']['buildinfo'], None)
    external_boxrender = compose_external_boxrender(packages['scheduler']['buildinfo'], Path(packages['scheduler']['prefix']), None,
        packages['kernels']['buildinfo'], packages['runner']['buildinfo'])
    # Compatibility metadata is composed from the verified canonical packages;
    # it never owns a second implementation or a child's payload.
    compatibility = selected.get("runtime_compat")
    if compatibility is not None:
        if compatibility.get("kind") != "package":
            raise ValueError("Compatibility metadata must be an installed package")
        compat_root = Path(compatibility["path"])
    else:
        compat_root = output / "runtime_compat"
        generator = Path(packages["scheduler"]["prefix"]) / "share/SimaScheduler/tools/compatibility_package.py"
        if not generator.is_file():
            raise ValueError("Selected Scheduler lacks the installed compatibility metadata producer")
        subprocess.run([sys.executable, str(generator), "--runner-package", packages["runner"]["package"],
                        "--scheduler-package", packages["scheduler"]["package"],
                        "--package-dir", str(compat_root)], check=True)
    compat_info = receipt(compat_root, "SimaRuntime")
    for child in ("runner", "scheduler"):
        consumed = compat_info.get(child + "_build", {})
        for field in ("package_sha256", "installed_payload_sha256"):
            if not consumed.get(field) or consumed[field] != packages[child]["buildinfo"].get(field):
                raise ValueError(f"Compatibility metadata consumed a different {child} {field}")
    packages["runtime_compat"] = {"kind": "composed-package", "package": str(compat_root),
        "prefix": str(package_prefix(compat_root)), "config": str(find_config(compat_root, "SimaRuntime")),
        "buildinfo": compat_info}
    verify_package_ownership(packages)
    if headers_only:
        headers = output / "headers/usr/include"
        if headers.parent.parent.exists():
            shutil.rmtree(headers.parent.parent)
        headers.mkdir(parents=True)
        for item in packages.values():
            source = Path(item["prefix"]) / "include"
            if source.is_dir():
                for header in source.rglob("*"):
                    target = headers / header.relative_to(source)
                    if target.exists() and header.is_file():
                        raise ValueError(f"Duplicate dependency header ownership: {header.relative_to(source)}")
                shutil.copytree(source, headers, dirs_exist_ok=True)
        return packages
    # A fresh dependency archive cache cannot accidentally redeploy old Internals.
    debs = output / "debs"
    if debs.exists():
        shutil.rmtree(debs)
    debs.mkdir()
    for component, item in packages.items():
        candidates = sorted(Path(item["package"]).glob("*.deb"))
        expected_name = DEBIAN_ROLES[component]
        if len(candidates) != 1:
            raise ValueError(f"{component} package must contain exactly one {expected_name} .deb: {item['package']}")
        for path in candidates:
            identity = subprocess.check_output(["dpkg-deb", "-f", str(path), "Package"], text=True).strip()
            if identity != expected_name:
                raise ValueError(f"Unexpected package {identity} in {component} package directory")
            verify_deb_receipt(path, PACKAGE_NAMES[component], item["buildinfo"], Path(item["prefix"]))
            version = subprocess.check_output(["dpkg-deb", "-f", str(path), "Version"], text=True).strip()
            subprocess.run(["dpkg", "--validate-version", version], check=True, capture_output=True)
            item["debian_version"] = version
            if (debs / path.name).exists():
                raise ValueError(f"Duplicate dependency package filename: {path.name}")
            shutil.copy2(path, debs / path.name)
    verify_debian_cohort(packages, debs)
    if external_runtimes:
        verified = compose_external_runtimes(packages['runner']['buildinfo'], debs)
        if verified != external_runtimes:
            raise ValueError('Selected external backend runtimes changed during preparation')
        packages.setdefault('external_dependencies', {})['RunnerBackends'] = verified
    if external_codecs is not None:
        verified = compose_external_codecs(packages['scheduler']['buildinfo'], debs)
        if verified != external_codecs:
            raise ValueError('Selected external Codecs cohort changed during preparation')
        packages.setdefault('external_dependencies', {})['SimaCodecs'] = verified
    if external_boxrender is not None:
        verified = compose_external_boxrender(packages['scheduler']['buildinfo'], Path(packages['scheduler']['prefix']), debs,
            packages['kernels']['buildinfo'], packages['runner']['buildinfo'])
        if verified != external_boxrender:
            raise ValueError('Selected external BoxRender cohort changed during preparation')
        packages.setdefault('external_dependencies', {})['BoxRender'] = verified
    verify_package_ownership(packages)
    for component, item in packages.items():
        if component in PACKAGE_NAMES and receipt(Path(item["package"]), PACKAGE_NAMES[component]) != item["buildinfo"]:
            raise ValueError(f"Selected {component} changed during final composition")
    cmake = output / "dependencies.cmake"
    def cache(name, value, kind="PATH"):
        # Bracket quoting preserves spaces and avoids CMake variable expansion.
        if "]==]" in value:
            raise ValueError("Unsupported CMake path delimiter in selected dependency")
        return f"set({name} [==[{value}]==] CACHE {kind} \"Selected local dependency\" FORCE)\n"
    content = cache("SIMANEAT_USE_EXPORTED_NEAT_INTERNALS", "ON", "BOOL")
    content += cache("SimaKernels_DIR", packages["kernels"]["config"])
    for component, item in packages.items():
        if component in PACKAGE_NAMES:
            content += cache(PACKAGE_NAMES[component] + "_DIR", item["config"])
    for component in (*COMPONENTS, "runtime_compat"):
        content += cache("SIMANEAT_" + component.upper() + "_DEBIAN_VERSION",
                         packages[component]["debian_version"], "STRING")
    # Runtime still exports this compatibility package for Core's platform ABI.
    content += cache("NeatInternals_DIR", str(find_config(Path(packages["runtime_compat"]["package"]), "NeatInternals")))
    cmake.write_text(content)
    (output / "resolved_manifest.json").write_text(json.dumps(packages, indent=2) + "\n")
    return packages


def verify_debian_cohort(packages: dict, debs: Path) -> None:
    """Bind SDK/runtime dependency edges to actual selected archive versions."""
    controls = {}
    for archive in sorted(debs.glob("*.deb")):
        fields = subprocess.check_output(["dpkg-deb", "-f", str(archive),
                                          "Package", "Architecture", "Depends"], text=True)
        control = dict(line.split(": ", 1) for line in fields.splitlines())
        role = control["Package"]
        if role in controls or control.get("Architecture") != "arm64":
            raise ValueError(f"Duplicate/wrong architecture dependency DEB: {archive}")
        controls[role] = control
    for consumer, dependencies in (("runner", ("kernels",)),
                                   ("scheduler", ("kernels", "runner")),
                                   ("runtime_compat", ("runner", "scheduler"))):
        terms = {term.strip() for term in controls[DEBIAN_ROLES[consumer]].get("Depends", "").split(",")}
        for dependency in dependencies:
            expected = f"{DEBIAN_ROLES[dependency]} (= {packages[dependency]['debian_version']})"
            if expected not in terms:
                raise ValueError(f"{consumer} DEB lacks exact selected dependency: {expected}")
    terms = {term.strip() for term in controls['simaai-runner'].get('Depends', '').split(',')}
    for provider in packages['runner']['buildinfo'].get('external_dependencies', {}).values():
        if not isinstance(provider, dict) or 'debian_provider' not in provider or 'loader_name' not in provider:
            continue
        runtime = provider['debian_provider']
        expected = f"{runtime['Package']} (= {runtime['Version']})"
        if expected not in terms:
            raise ValueError(f'Runner DEB lacks exact selected backend dependency: {expected}')
    renderer = packages['scheduler']['buildinfo'].get('external_dependencies', {}).get('BoxRender')
    if renderer:
        runtime = renderer['packages']['runtime']
        expected = f"{runtime['Package']} (= {runtime['Version']})"
        terms = {term.strip() for term in controls['simaai-scheduler'].get('Depends', '').split(',')}
        if expected not in terms:
            raise ValueError(f'Scheduler DEB lacks exact selected BoxRender dependency: {expected}')


def verify_package_ownership(packages: dict) -> None:
    owners = {}
    for component, item in packages.items():
        if component not in PACKAGE_NAMES:
            continue
        prefix = Path(item["prefix"])
        for path in prefix.rglob("*"):
            if not (path.is_file() or path.is_symlink()):
                continue
            name = path.relative_to(prefix).as_posix()
            if name in owners:
                raise ValueError(f"Duplicate dependency payload owner for {name}: {owners[name]} and {component}")
            owners[name] = component
    codecs = packages.get("external_dependencies", {}).get("SimaCodecs")
    if codecs:
        for name in codec_prefix_inventory(Path(codecs["prefix"])):
            local = name.removeprefix("usr/")
            if local in owners:
                raise ValueError(f"Codecs duplicates dependency payload ownership: {local}")
            owners[local] = 'codecs'
    boxrender = packages.get('external_dependencies', {}).get('BoxRender')
    if boxrender:
        for name in boxrender['buildinfo']['installed_files']:
            # Component payloads above are relative to /usr; the provider may
            # also own /opt configuration assets, outside those namespaces.
            if name.startswith('usr/') and name.removeprefix('usr/') in owners:
                raise ValueError(f'BoxRender duplicates dependency payload ownership: {name}')
            owners[name.removeprefix('usr/')] = 'BoxRender'
    external_owners = {}
    for backend, item in packages.get('external_dependencies', {}).get('RunnerBackends', {}).items():
        role = item['debian_provider']['Package']
        for name in item['installed_files']:
            if name.startswith('usr/') and name.removeprefix('usr/') in owners:
                raise ValueError(f'{backend} duplicates dependency payload ownership: {name}')
            previous = external_owners.setdefault(name, role)
            if previous != role:
                raise ValueError(f'Backend runtime packages overlap ownership: {name}')


def verify_deb_receipt(deb: Path, name: str, expected: dict, prefix: Path) -> None:
    """Audit exact installed files, full permission bits and canonical ownership."""
    receipt_path = f'usr/share/{name}/buildinfo.json'
    found = None;expected_files = {};packaged_files = {};seen = set()
    for path in prefix.rglob('*'):
        relative = 'usr/' + path.relative_to(prefix).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            if os.path.isabs(target) or not path.resolve().is_relative_to(prefix):
                raise ValueError(f'Non-relocatable installed dependency link: {relative}')
            expected_files[relative] = ('link', target)
        elif path.is_file():
            expected_files[relative] = ('file', hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mode & 0o7777)
    # Runtime's canonical make_deb adds its loader configuration outside /usr.
    # It is not part of the installed SDK prefix: allow this one exact contract,
    # never arbitrary /etc files. Its bytes/mode must match the producer.
    loader = 'etc/ld.so.conf.d/sima-neat.conf'
    loader_dirs = {'etc', 'etc/ld.so.conf.d'}
    loader_required = name == 'SimaRunner' and (prefix/'lib/aarch64-linux-gnu/neat/runtime').is_dir()
    if loader_required:
        data = b'/usr/lib/aarch64-linux-gnu/neat/runtime\n/usr/lib/aarch64-linux-gnu/neat/gst-plugins\n'
        expected_files[loader] = ('file', hashlib.sha256(data).hexdigest(), 0o644)
    with dependency_archive(deb) as archive:
        for member in archive:
            relative = dependency_archive_path(member)
            if relative in seen:
                raise ValueError(f'Duplicate dependency archive entry: {relative}')
            seen.add(relative)
            if relative == '':
                continue
            if relative not in ('usr',) and not relative.startswith('usr/'):
                if not loader_required or (relative != loader and relative not in loader_dirs):
                    raise ValueError(f'Dependency archive payload outside owned paths: {relative}')
                if relative in loader_dirs:
                    if not member.isdir() or member.mode & 0o7777 != 0o755:
                        raise ValueError('Runtime loader parent directory differs from producer contract')
                    continue
            if member.isfile():
                data = archive.extractfile(member).read()
                packaged_files[relative] = ('file', hashlib.sha256(data).hexdigest(), member.mode & 0o7777)
                if relative == receipt_path:
                    found = json.loads(data)
            elif member.issym():
                packaged_files[relative] = ('link', member.linkname)
            elif member.islnk():
                target = dependency_archive_path(tarfile.TarInfo(member.linkname))
                if target not in packaged_files or packaged_files[target][0] != 'file':
                    raise ValueError(f'Unsupported forward/nonfile hard link in {deb}: {relative}')
                packaged_files[relative] = ('file', packaged_files[target][1], member.mode & 0o7777)
            elif not member.isdir():
                raise ValueError(f'Unsupported dependency archive entry: {relative}')
    if packaged_files != expected_files:
        raise ValueError(f'{deb} payload does not match the selected installed /usr tree and owned loader contract; rebuild the package.')
    if found != expected:
        raise ValueError(f'{deb} does not match the selected installed package receipt; rebuild the package.')


def validate_cohort(packages: dict) -> None:
    kernels = packages["kernels"]["buildinfo"]
    runner = packages["runner"]["buildinfo"]
    # Schema supplied by the standalone package producers; never use source
    # revision equality as a substitute for the consumed contract identity.
    header_root = Path(packages["kernels"]["prefix"]) / "include"
    entries = [f"{path.relative_to(header_root).as_posix()}:{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
               for path in sorted(header_root.rglob("*"))
               if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"]
    if not entries or hashlib.sha256("".join(entries).encode()).hexdigest() != kernels.get("contract_sha256"):
        raise ValueError("Installed Kernels headers do not match the package contract receipt; rebuild the package.")
    expected = runner.get("kernels_contract_sha256")
    actual = kernels.get("contract_sha256")
    if not expected or not actual or expected != actual:
        raise ValueError("Runner was compiled against a different or unrecorded Kernels contract. Rebuild Runner against the selected Kernels package.")
    consumed = runner.get("kernels_build", {})
    if kernels.get("schema_version") == 2:
        for field in ("package_sha256", "installed_payload_sha256"):
            if not kernels.get(field) or kernels[field] != consumed.get(field):
                raise ValueError(f"Runner consumed a different Kernels {field}; rebuild Runner against the selected SDK.")
    for field in ("flags", "build_type"):
        if field not in consumed or kernels.get(field) != consumed[field]:
            raise ValueError(f"Runner consumed different Kernels {field}; rebuild Runner.")
    if kernels.get("toolchain", {}).get("sha256") != consumed.get("toolchain", {}).get("sha256"):
        raise ValueError("Runner consumed a different Kernels toolchain; rebuild Runner.")
    for field in ("architecture", "build_type", "sysroot"):
        if field not in kernels or kernels.get(field) != runner.get(field):
            raise ValueError(f"Kernels/Runner {field} mismatch. Rebuild both with the same target/toolchain.")
    for group, fields in (("compiler", ("sha256", "target")),):
        for field in fields:
            if not kernels.get(group, {}).get(field) or kernels.get(group, {}).get(field) != runner.get(group, {}).get(field):
                raise ValueError(f"Kernels/Runner {group}.{field} mismatch. Rebuild both with the same target/toolchain.")


    scheduler = packages["scheduler"]["buildinfo"]
    consumed = scheduler.get("runner_build", {})
    for field in ("package_sha256", "installed_payload_sha256", "contract_sha256"):
        if not runner.get(field) or runner[field] != consumed.get(field):
            raise ValueError(f"Scheduler consumed a different or unrecorded Runner {field}; rebuild Scheduler.")
    runner_headers = Path(packages["runner"]["prefix"]) / "include"
    entries = [f"{path.relative_to(runner_headers).as_posix()}:{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
               for path in sorted(runner_headers.rglob("*")) if path.is_file()]
    if not entries or hashlib.sha256("".join(entries).encode()).hexdigest() != runner.get("contract_sha256"):
        raise ValueError("Installed Runner headers do not match the package contract receipt")
    for field in ("architecture", "build_type", "sysroot"):
        if field not in runner or scheduler.get(field) != runner[field]:
            raise ValueError(f"Runner/Scheduler {field} mismatch")
    for group, fields in (("compiler", ("sha256", "target")), ("toolchain", ("sha256",))):
        for field in fields:
            if not runner.get(group, {}).get(field) or scheduler.get(group, {}).get(field) != runner[group][field]:
                raise ValueError(f"Runner/Scheduler {group}.{field} mismatch")
    # Scheduler's transitive numerical closure must use the exact selected SDK,
    # not just a header-compatible archive from another build.
    for field in ("package_sha256", "installed_payload_sha256", "flags", "build_type", "toolchain"):
        if field in kernels and scheduler.get("kernels_build", {}).get(field) != kernels[field]:
            raise ValueError(f"Scheduler consumed different Kernels {field}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "prepare"))
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--headers-only", action="store_true")
    parser.add_argument("--require", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    try:
        selected = select(args.core.resolve(), dict(os.environ))
        if args.command == "select":
            if args.require:
                for component in COMPONENTS:
                    if component not in selected:
                        raise ValueError(f"Missing local {component}; set NEAT_{component.upper()}_SOURCE or NEAT_{component.upper()}_PACKAGE_DIR. TODO: remote artifact pulling.")
                    if args.headers_only and selected[component]["kind"] == "source":
                        raise ValueError(f"Header-only builds do not compile {component}; provide NEAT_{component.upper()}_PACKAGE_DIR. TODO: remote artifact pulling.")
            for component, item in selected.items():
                suffix = "SOURCE" if item["kind"] == "source" else "PACKAGE_DIR"
                print(f"export NEAT_{component.upper()}_{suffix}={shlex.quote(item['path'])}")
        else:
            if not args.output:
                parser.error("prepare requires --output")
            prepare(args.core.resolve(), args.output.resolve(), selected, args.jobs, args.headers_only, args.clean)
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
