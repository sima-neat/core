"""Actual host ELF probes for the installed-package test's loaded-provider boundary."""
from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import platform
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('inventory_verifier', ROOT / 'tools/ci/verify_core_dist.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
SONAME = 'libinventory_fixture.so.1'


def host_architecture():
    machine = platform.machine().lower()
    identities = {'x86_64': ('x86-64', 'amd64'), 'amd64': ('x86-64', 'amd64'),
                  'aarch64': ('ARM aarch64', 'arm64'), 'arm64': ('ARM aarch64', 'arm64')}
    assert machine in identities, f'unsupported host ELF test architecture: {machine}'
    return identities[machine]


def inspect_host_elf(path):
    # This suite builds native host probes only, never SDK/board artifacts.
    # The x86 SDK must never execute an AArch64 build; native ARM CI may use
    # its own native compiler without pretending the compiler is x86.
    description = subprocess.check_output(['file', str(path)], text=True)
    assert 'ELF' in description and host_architecture()[0] in description, description


@pytest.fixture(scope='module')
def elf_probe(tmp_path_factory):
    for tool in ('c++', 'readelf', 'file', 'dpkg-deb'):
        assert shutil.which(tool), f'required ELF inventory test tool is missing: {tool}'
    root = tmp_path_factory.mktemp('inventory-elf')
    source = root / 'provider.cpp'
    source.write_text('extern "C" int inventory_fixture_value() { return 17; }\n')
    selected = root / 'libinventory_fixture.so.1.NEW'
    stale = root / 'libinventory_fixture.so.1.OLD'
    for path in (selected, stale):
        subprocess.run(['c++', '-shared', '-fPIC', str(source), '-Wl,-soname,' + SONAME,
                        '-o', str(path)], check=True, capture_output=True)
        inspect_host_elf(path)
    probe = root / 'probe.cpp'
    probe.write_text(r'''
#include "support/package_loaded_identity.h"
#include <array>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <unistd.h>
using namespace simaai::neat::test_support;

void bounds_probe() {
  // Both supported loader representations must read the same bounded bytes.
  std::array<ElfW(Addr), 64> bytes{};
  auto base = reinterpret_cast<std::uintptr_t>(bytes.data());
  std::array<ElfW(Phdr), 2> headers{};
  headers[0].p_type = PT_LOAD;
  headers[0].p_flags = PF_R;
  headers[0].p_memsz = sizeof(bytes);
  headers[1].p_type = PT_DYNAMIC;
  headers[1].p_memsz = 4 * sizeof(ElfW(Dyn));
  dl_phdr_info info{};
  info.dlpi_addr = base;
  info.dlpi_phdr = headers.data();
  info.dlpi_phnum = headers.size();
  auto* entries = reinterpret_cast<ElfW(Dyn)*>(bytes.data());
  entries[0].d_tag = DT_STRTAB;
  entries[1].d_tag = DT_STRSZ;
  entries[1].d_un.d_val = 10;
  entries[2].d_tag = DT_SONAME;
  entries[2].d_un.d_val = 0;
  entries[3].d_tag = DT_NULL;
  std::memcpy(reinterpret_cast<void*>(base + 128), "libx.so.1", 10);
  for (auto strings : {base + 128, std::uintptr_t{128}}) {
    entries[0].d_un.d_ptr = strings;
    if (loaded_identity_detail::loaded_soname(info) != "libx.so.1")
      throw std::runtime_error("absolute/relative DT_STRTAB mismatch");
  }
  auto reject = [&] {
    try { (void)loaded_identity_detail::loaded_soname(info); }
    catch (const std::runtime_error&) { return; }
    throw std::runtime_error("unbounded dynamic metadata was accepted");
  };
  entries[0].d_tag = DT_DEBUG;
  reject();
  entries[0].d_tag = DT_STRTAB;
  entries[1].d_tag = DT_DEBUG;
  reject();
  entries[1].d_tag = DT_STRSZ;
  entries[2].d_un.d_val = 10;
  reject();
  entries[2].d_un.d_val = 0;
  entries[1].d_un.d_val = sizeof(bytes) + 1;
  reject();
  entries[1].d_un.d_val = 9; // no terminating null within DT_STRSZ
  reject();
  entries[1].d_un.d_val = 10;
  entries[0].d_un.d_ptr = sizeof(bytes) + 128;
  reject();
  entries[0].d_un.d_ptr = 128;
  headers[1].p_memsz = sizeof(bytes) + 1;
  reject();
}

int main(int argc, char** argv) {
  try {
    if (argc == 1) { bounds_probe(); return 0; }
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(dlerror());
    auto value = reinterpret_cast<int (*)()>(dlsym(library, "inventory_fixture_value"));
    if (!value || value() != 17) throw std::runtime_error("fixture did not execute");
    if (argc >= 5 && unlink(argv[1]) != 0) throw std::runtime_error("fixture unlink failed");
    if (argc == 6) std::filesystem::copy_file(argv[5], argv[1]);
    auto loaded = selected_loaded_libraries({{argv[2], {argv[3]}}});
    if (loaded.size() != 1 || loaded.front().path != argv[3])
      throw std::runtime_error("selected actual DSO not observed");
    dlclose(library);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
''')
    binary = root / 'probe'
    subprocess.run(['c++', '-std=c++20', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'tests'),
                    str(probe), '-ldl', '-o', str(binary)], check=True, capture_output=True)
    inspect_host_elf(binary)
    return root, binary, selected, stale


def test_real_loaded_selected_soname_passes(elf_probe):
    _, binary, selected, _ = elf_probe
    inspect_host_elf(binary)
    subprocess.run([str(binary), str(selected), SONAME, str(selected)], check=True, capture_output=True)


def test_same_soname_different_real_filename_is_rejected(elf_probe):
    _, binary, selected, stale = elf_probe
    inspect_host_elf(binary)
    result = subprocess.run([str(binary), str(stale), SONAME, str(selected)], text=True, capture_output=True)
    assert result.returncode == 1
    assert SONAME in result.stderr and str(stale) in result.stderr
    assert 'outside its installed package' in result.stderr


@pytest.mark.parametrize('replace_file', [False, True])
def test_deleted_mapping_is_rejected_even_at_expected_path(elf_probe, tmp_path, replace_file):
    _, binary, selected, _ = elf_probe
    removed = tmp_path / selected.name
    shutil.copyfile(selected, removed)
    inspect_host_elf(removed)
    inspect_host_elf(binary)
    command = [str(binary), str(removed), SONAME, str(removed), 'unlink']
    if replace_file:
        command.append(str(selected))
    result = subprocess.run(command, text=True, capture_output=True)
    assert removed.exists() == replace_file
    assert result.returncode == 1
    assert ' (deleted)' in result.stderr and 'outside its installed package' in result.stderr


def test_loaded_dynamic_metadata_reads_are_bounded(elf_probe):
    _, binary, _, _ = elf_probe
    inspect_host_elf(binary)
    subprocess.run([str(binary)], check=True, capture_output=True)


def test_payload_identity_comes_from_elf_not_its_filename(elf_probe, tmp_path):
    _, _, selected, _ = elf_probe
    source = tmp_path / 'package'
    (source / 'DEBIAN').mkdir(parents=True)
    (source / 'DEBIAN/control').write_text(
        f'Package: inventory-elf-fixture\nVersion: 1\nArchitecture: {host_architecture()[1]}\n'
        'Maintainer: Fixture <fixture@example.invalid>\nDescription: actual ELF inventory fixture\n')
    payload = source / 'usr/lib'; payload.mkdir(parents=True)
    shutil.copyfile(selected, payload / 'arbitrary-provider-filename')
    (payload / SONAME).symlink_to('arbitrary-provider-filename')
    archive = tmp_path / 'provider.deb'
    subprocess.run(['dpkg-deb', '--build', '--root-owner-group', str(source), str(archive)],
                   check=True, capture_output=True)
    inventory = verifier.deb_payload(archive)
    identity = inventory['/usr/lib/arbitrary-provider-filename']
    assert identity['soname'] == SONAME
    assert identity['sha256'] == hashlib.sha256(selected.read_bytes()).hexdigest()
    assert 'soname' not in inventory['/usr/lib/' + SONAME]  # link is not a second ELF


def test_non_elf_payload_needs_no_binutils(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('non-ELF file invoked binutils')
    monkeypatch.setattr(verifier.subprocess, 'check_output', unexpected)
    digest = hashlib.sha256()
    assert verifier.payload_soname(io.BytesIO(b'plain data'), digest) is None
    assert digest.hexdigest() == hashlib.sha256(b'plain data').hexdigest()


def test_elf_payload_requires_binutils(elf_probe, monkeypatch):
    _, _, selected, _ = elf_probe
    def unavailable(*args, **kwargs):
        raise FileNotFoundError('readelf missing')
    monkeypatch.setattr(verifier.subprocess, 'check_output', unavailable)
    with selected.open('rb') as stream, pytest.raises(FileNotFoundError, match='readelf missing'):
        verifier.payload_soname(stream, hashlib.sha256())
