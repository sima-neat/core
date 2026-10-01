#!/usr/bin/env python3
"""Hermetic package-cohort regression tests; inspect ELFs, never execute them."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "tools" / "install_neat_framework.sh"


class BundleCohortTest(unittest.TestCase):
    def setUp(self):
        for tool in ("cc", "readelf", "dpkg-deb", "bash"):
            if not shutil.which(tool):
                self.fail(f"required test tool missing: {tool}")
        self.temp = tempfile.TemporaryDirectory(prefix="cohort-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0
        self.arch = subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()
        if self.arch not in ("amd64", "arm64"):
            self.fail(f"unsupported fixture compiler architecture: {self.arch}")

    def library(self, name, soname, source, dependencies=(), rpath=None, linker_flags=()):
        self.serial += 1
        directory = self.root / f"elf-{self.serial}"
        directory.mkdir()
        path = directory / name
        src = directory / "fixture.c"
        src.write_text(source)
        subprocess.run(["cc", "-shared", "-fPIC", "-nostdlib", f"-Wl,-soname,{soname}",
                        str(src), *map(str, dependencies), *linker_flags,
                        *([f"-Wl,-rpath,{rpath}"] if rpath is not None else []),
                        "-o", str(path)], check=True)
        return path

    def package(self, package, version="0.4.0", payload=(), architecture=None, install_dir="usr/lib/neat-test", directory_links=()):
        self.serial += 1
        root = self.root / f"package-{self.serial}"
        (root / "DEBIAN").mkdir(parents=True)
        (root / "DEBIAN/control").write_text(
            f"Package: {package}\nVersion: {version}\nArchitecture: {architecture or self.arch}\n"
            "Maintainer: Cohort test <test@example.invalid>\nDescription: Test fixture\n")
        for source in payload:
            target = root / install_dir / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                target.symlink_to(source.readlink())
            else:
                shutil.copyfile(source, target)
        for name, destination in directory_links:
            link = root / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(destination)
        deb = self.root / f"fixture-{self.serial}.deb"
        subprocess.run(["dpkg-deb", "--build", "--root-owner-group", str(root), str(deb)],
                       check=True, stdout=subprocess.DEVNULL)
        return deb

    def check_bundle(self, *debs):
        return subprocess.run(["bash", "-c",
                               'set -euo pipefail; source "$1"; shift; DEBS=("$@"); validate_bundle_elf_cohort',
                               "bash", str(INSTALLER), *map(str, debs)], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def pair(self, needed="0.4.0"):
        old = self.library(f"libneatdispatchercore.so.{needed}", f"libneatdispatchercore.so.{needed}",
                           "int dispatcher(void) { return 1; }")
        current = old if needed == "0.4.0" else self.library(
            "libneatdispatchercore.so.0.4.0", "libneatdispatchercore.so.0.4.0",
            "int dispatcher(void) { return 1; }")
        consumer = self.library("libsima_lmm_runtime.so", "libsima_lmm_runtime.so",
                                "extern int dispatcher(void); int runtime(void) { return dispatcher(); }", [old], rpath="$ORIGIN")
        return self.package("neat-runtime", payload=[current]), self.package("sima-lmm-core", payload=[consumer])

    def test_matching_selected_dependencies(self):
        result = self.check_bundle(*self.pair())
        self.assertEqual(result.returncode, 0, result.stdout)

    def split_pair(self, rpath=None, provider_dir="usr/lib/aarch64-linux-gnu/neat/runtime"):
        provider = self.library("libneatdispatchercore.so.0.4.0", "libneatdispatchercore.so.0.4.0",
                                "int dispatcher(void) { return 1; }")
        consumer = self.library("libgstneatfixture.so", "libgstneatfixture.so",
                                "extern int dispatcher(void); int plugin(void) { return dispatcher(); }",
                                [provider], rpath=rpath)
        return (self.package("neat-runtime", payload=[provider], install_dir=provider_dir),
                self.package("neat-gst-plugins", payload=[consumer],
                             install_dir="usr/lib/aarch64-linux-gnu/neat/gst-plugins"))

    def test_rejects_unreachable_private_provider(self):
        for rpath in (None, "$ORIGIN", "$ORIGIN/../wrong"):
            with self.subTest(rpath=rpath):
                result = self.check_bundle(*self.split_pair(rpath))
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("not reachable", result.stdout)

    def test_accepts_consumer_private_runpath(self):
        for rpath in ("$ORIGIN/../runtime", "${ORIGIN}/../runtime",
                      "/usr/lib/aarch64-linux-gnu/neat/runtime"):
            with self.subTest(rpath=rpath):
                result = self.check_bundle(*self.split_pair(rpath))
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_accepts_default_library_directory(self):
        result = self.check_bundle(*self.split_pair(provider_dir="usr/lib"))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_accepts_packaged_loader_configuration(self):
        config = self.root / "sima-neat.conf"
        config.write_text("# Private NEAT runtime roots\n"
                          "/usr/lib/aarch64-linux-gnu/neat/runtime\n")
        result = self.check_bundle(
            *self.split_pair(),
            self.package("neat-common", payload=[config], install_dir="etc/ld.so.conf.d"))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_rejects_missing_consumer_runpath_in_transitive_chain(self):
        leaf = self.library("libneatleaf.so.1", "libneatleaf.so.1", "int leaf(void) { return 1; }")
        middle = self.library("libneatmiddle.so.1", "libneatmiddle.so.1",
                              "extern int leaf(void); int middle(void) { return leaf(); }", [leaf])
        top = self.library("libsima_neat.so.5", "libsima_neat.so.5",
                           "extern int middle(void); int top(void) { return middle(); }",
                           [middle], rpath="$ORIGIN/../neat-test")
        result = self.check_bundle(
            self.package("neat-runtime", payload=[leaf, middle]),
            self.package("sima-neat", payload=[top], install_dir="usr/lib/top"))
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("not reachable", result.stdout)

    def test_accepts_direct_legacy_rpath(self):
        provider = self.library("libneatfixture.so.1", "libneatfixture.so.1", "int value;")
        consumer = self.library("libsima_neat.so.5", "libsima_neat.so.5",
                                "extern int value; int result(void) { return value; }",
                                [provider], rpath="$ORIGIN", linker_flags=["-Wl,--disable-new-dtags"])
        result = self.check_bundle(self.package("neat-runtime", payload=[provider, consumer]))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_default_library_paths_are_architecture_specific(self):
        native = "x86_64-linux-gnu" if self.arch == "amd64" else "aarch64-linux-gnu"
        foreign = "aarch64-linux-gnu" if self.arch == "amd64" else "x86_64-linux-gnu"
        for directory, accepted in ((native, True), (foreign, False)):
            with self.subTest(directory=directory):
                result = self.check_bundle(*self.split_pair(provider_dir=f"usr/lib/{directory}"))
                self.assertEqual(result.returncode == 0, accepted, result.stdout)

    def test_nodefaultlib_does_not_accept_default_provider(self):
        provider = self.library("libneatfixture.so.1", "libneatfixture.so.1", "int value;")
        for rpath, accepted in ((None, False), ("/usr/lib", True)):
            with self.subTest(rpath=rpath):
                consumer = self.library("libsima_neat.so.5", "libsima_neat.so.5",
                                        "extern int value; int result(void) { return value; }",
                                        [provider], rpath=rpath, linker_flags=["-Wl,-z,nodefaultlib"])
                result = self.check_bundle(
                    self.package("neat-runtime", payload=[provider], install_dir="usr/lib"),
                    self.package("sima-neat", payload=[consumer]))
                self.assertEqual(result.returncode == 0, accepted, result.stdout)

    def test_reachable_soname_symlink(self):
        provider = self.library("libneatfixture.so.1.2.3", "libneatfixture.so.1", "int value;")
        link = provider.parent / "libneatfixture.so.1"
        link.symlink_to(provider.name)
        consumer = self.library("libsima_neat.so.5", "libsima_neat.so.5",
                                "extern int value; int result(void) { return value; }",
                                [provider], rpath="$ORIGIN")
        result = self.check_bundle(self.package("neat-runtime", payload=[provider, link, consumer]))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_matching_soname_symlink(self):
        lib = self.library("libneatdispatchercore.so.0.4.0", "libneatdispatchercore.so.0", "int value;")
        link = lib.parent / "libneatdispatchercore.so.0"
        link.symlink_to(lib.name)
        result = self.check_bundle(self.package("neat-runtime", payload=[lib, link]))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_rejects_build_rpath(self):
        lib = self.library("libsima_neat.so.5", "libsima_neat.so.5", "int value;", rpath="/repair/stale/lib")
        result = self.check_bundle(self.package("sima-neat", payload=[lib]))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsafe build/empty runtime search path", result.stdout)

    def test_rejects_empty_or_cwd_relative_rpath(self):
        for rpath in ("", "relative/lib", "$ORIGIN:", "$ORIGIN_NOT_A_TOKEN/lib"):
            with self.subTest(rpath=rpath):
                lib = self.library("libsima_neat.so.5", "libsima_neat.so.5", "int value;", rpath=rpath)
                result = self.check_bundle(self.package("sima-neat", payload=[lib]))
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("unsafe build/empty runtime search path", result.stdout)

    def test_rejects_normalized_runtime_path_escapes(self):
        for rpath in ("$ORIGIN/../../../../tmp/stale",
                      "${ORIGIN}/../../../home/stale", "/usr/lib/../../tmp/stale",
                      "$ORIGIN/../../../opt/toolchain/lib", "$ORIGIN/../../../var/stale",
                      "/usr/library/stale", "$ORIGIN/$UNKNOWN"):
            with self.subTest(rpath=rpath):
                lib = self.library("libsima_neat.so.5", "libsima_neat.so.5", "int value;", rpath=rpath)
                result = self.check_bundle(self.package("sima-neat", payload=[lib]))
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("unsafe build/empty runtime search path", result.stdout)

    def test_accepts_normalized_runtime_paths(self):
        for rpath in ("$ORIGIN", "${ORIGIN}", "$ORIGIN/../neat/runtime",
                      "${ORIGIN}/../aarch64-linux-gnu/neat/runtime",
                      "/usr/lib/aarch64-linux-gnu/neat/runtime", "/usr/lib/../lib"):
            with self.subTest(rpath=rpath):
                lib = self.library("libsima_neat.so.5", "libsima_neat.so.5", "int value;", rpath=rpath)
                result = self.check_bundle(self.package("sima-neat", payload=[lib]))
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_rejects_original_dispatcher_mismatch(self):
        result = self.check_bundle(*self.pair("1"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires libneatdispatchercore.so.1", result.stdout)

    def test_rejects_missing_gst_provider(self):
        provider = self.library("libgstneattensorbuffer.so.0.4.0", "libgstneattensorbuffer.so.0.4.0",
                                "int tensor(void) { return 1; }")
        consumer = self.library("libsima_neat.so.5", "libsima_neat.so.5",
                                "extern int tensor(void); int core(void) { return tensor(); }", [provider])
        result = self.check_bundle(self.package("sima-neat", payload=[consumer]))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires libgstneattensorbuffer", result.stdout)

    def test_rejects_mixed_package_architectures(self):
        other_arch = "arm64" if self.arch == "amd64" else "amd64"
        result = self.check_bundle(self.package("neat-runtime"),
                                   self.package("sima-neat", architecture=other_arch))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("package architecture", result.stdout)

    def test_rejects_duplicate_package(self):
        package = self.package("neat-runtime")
        result = self.check_bundle(package, package)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate package", result.stdout)

    def test_rejects_mixed_internals_versions(self):
        result = self.check_bundle(self.package("neat-runtime", "0.4.0"), self.package("neat-common", "0.5.0"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mixed internals versions", result.stdout)

    def test_rejects_mixed_llima_versions(self):
        result = self.check_bundle(self.package("sima-lmm-core", "0.4.0"), self.package("sima-lmm-dev", "0.5.0"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mixed llima versions", result.stdout)

    def test_rejects_missing_soname_path(self):
        lib = self.library("libneatdispatchercore.so.0.4.0", "libneatdispatchercore.so.1", "int value;")
        result = self.check_bundle(self.package("neat-runtime", payload=[lib]))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SONAME path", result.stdout)

    def test_rejects_wrong_architecture(self):
        lib = self.library("libneatdispatchercore.so.1", "libneatdispatchercore.so.1", "int value;")
        other_arch = "arm64" if self.arch == "amd64" else "amd64"
        result = self.check_bundle(self.package("neat-runtime", payload=[lib], architecture=other_arch))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("architecture mismatch", result.stdout)

    def test_rejects_conflicting_payloads(self):
        a = self.library("libneatdispatchercore.so.1", "libneatdispatchercore.so.1", "int value = 1;")
        b = self.library("libneatdispatchercore.so.1", "libneatdispatchercore.so.1", "int value = 2;")
        result = self.check_bundle(self.package("neat-runtime", payload=[a]), self.package("neat-common", payload=[b]))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("conflicting payload", result.stdout)

    def test_rejects_directory_symlink_payload_conflict(self):
        provider = self.library("libneatfixture.so.1", "libneatfixture.so.1", "int value;")
        linked = self.package("neat-runtime", payload=[provider],
                              directory_links=[("usr/lib/alias", "neat-test")])
        directory = self.package("neat-common", payload=[provider], install_dir="usr/lib/alias")
        for packages in ((linked, directory), (directory, linked)):
            with self.subTest(order=packages):
                result = self.check_bundle(*packages)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("conflicting payload path usr/lib/alias", result.stdout)

    def test_rejects_directory_file_payload_conflict(self):
        file = self.root / "alias"
        file.write_text("not a directory")
        child = self.root / "child"
        child.write_text("nested file")
        directory = self.package("neat-runtime", payload=[child], install_dir="usr/lib/alias")
        regular = self.package("neat-common", payload=[file], install_dir="usr/lib")
        for packages in ((directory, regular), (regular, directory)):
            with self.subTest(order=packages):
                result = self.check_bundle(*packages)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("conflicting payload path usr/lib/alias", result.stdout)

    def test_accepts_shared_directories_and_identical_directory_links(self):
        child = self.root / "child"
        child.write_text("shared data")
        arguments = dict(payload=[child], directory_links=[("usr/lib/alias", "neat-test")])
        result = self.check_bundle(self.package("neat-runtime", **arguments),
                                   self.package("neat-common", **arguments))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_preflight_is_before_install_mutations(self):
        text = INSTALLER.read_text()
        main = text[text.rindex('parse_args "$@"'):]
        self.assertLess(main.index("validate_bundle_elf_cohort"), main.index("install_for_environment"))


if __name__ == "__main__":
    unittest.main()
