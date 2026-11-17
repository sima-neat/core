from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASH_VERSION = subprocess.check_output(
    ["bash", "-c", 'printf "%s" "${BASH_VERSINFO[0]}"'], text=True
)


@unittest.skipUnless(int(BASH_VERSION) >= 4, "The tidy script requires Bash 4+")
class CppTidyScopeTests(unittest.TestCase):
    def run_tidy(
        self, changed: str, *, install: bool = True, mode: str = "changed", failure: int = 0, configure_failure: int = 0, sdk: bool = False
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts/ci").mkdir(parents=True)
            shutil.copyfile(ROOT / "scripts/run_cpp_tidy.sh", root / "scripts/run_cpp_tidy.sh")
            for folder in ("include", "src", "tutorials", "examples", "tests"):
                (root / folder).mkdir()
            (root / "src/sample.cpp").write_text("int sample() { return 0; }\n")
            (root / "README.md").write_text("Documentation\n")
            (root / "include/sample.h").write_text("int sample();\n")
            log = root / "calls"
            bin_dir = root / "bin"
            bin_dir.mkdir()
            native_variables = ("SYSROOT", "SDKTARGETSYSROOT", "CMAKE_TOOLCHAIN_FILE", "OE_CMAKE_TOOLCHAIN_FILE",
                                "CC", "CXX", "CXXFLAGS", "CPATH", "LD_LIBRARY_PATH", "PKG_CONFIG_LIBDIR")
            native_probe = "".join("${" + name + "}" for name in native_variables)
            commands = {
                "git": 'if [ "$1" = diff ]; then printf "%s\\n" "$TIDY_TEST_CHANGED"; fi',
                "rg": 'case " $* " in *" --files "*) printf "%s\\n" "$PWD/src/sample.cpp";; *) exit 1;; esac',
                "cmake": ('echo cmake >> "$TIDY_TEST_CALLS"; '
                          f'test -z "{native_probe}" || exit 91; '
                          'test "$CMAKE_PREFIX_PATH" = "$TIDY_TEST_PREFIX" || exit 92; '
                          'case " $* " in *" -DCMAKE_DISABLE_FIND_PACKAGE_SimaLMM=TRUE "*) :;; *) exit 93;; esac; '
                          f'exit {configure_failure}'),
                "clang-tidy": ":",
                "run-clang-tidy": 'echo tidy >> "$TIDY_TEST_CALLS"',
            }
            for name, body in commands.items():
                target = bin_dir / name
                target.write_text("#!/bin/sh\n" + body + "\n")
                target.chmod(0o755)
            bootstrap = root / "scripts/ci/install_code_hygiene_deps.sh"
            bootstrap.write_text(
                '#!/bin/sh\necho bootstrap >> "$TIDY_TEST_CALLS"\n'
                + f"exit {failure}\n"
            )
            bootstrap.chmod(0o755)
            # The SDK-aware build driver must never be invoked by analysis.
            (root / "build.sh").write_text('#!/bin/sh\necho WRONG_SDK_DRIVER >> "$TIDY_TEST_CALLS"\nexit 99\n')
            (root / "build.sh").chmod(0o755)
            scheduler = root / "selected-scheduler"
            support = root / "selected-test-support"
            configurations = [scheduler / "usr/lib/cmake/SimaScheduler/SimaSchedulerConfig.cmake",
                              support / "usr/lib/cmake/SimaSchedulerClientTestSupport/SimaSchedulerClientTestSupportConfig.cmake"]
            for config in configurations:
                config.parent.mkdir(parents=True)
                config.touch()
            prefix = str(root / "host-prefix")
            env = dict(os.environ)
            env.update(
                PATH=str(bin_dir) + os.pathsep + env["PATH"],
                TIDY_TEST_CALLS=str(log),
                TIDY_TEST_CHANGED=changed,
                NEAT_SCHEDULER_PACKAGE_DIR=str(scheduler),
                NEAT_SCHEDULER_CLIENT_TEST_PACKAGE_DIR=str(support),
                CMAKE_PREFIX_PATH=prefix,
                TIDY_TEST_PREFIX=os.pathsep.join([str(config.parent) for config in reversed(configurations)] + [prefix]),
                TIDY_INSTALL_DEPS="1" if install else "0",
                TIDY_BASE_REF="fixture-base",
                TIDY_JOBS="1",
            )
            if sdk:
                poison = root / "sdk/usr/lib/aarch64-linux-gnu/cmake/SimaLMM"
                poison.mkdir(parents=True)
                (poison / "SimaLMMConfig.cmake").write_text('message(FATAL_ERROR "ambient SDK")\n')
                for key in native_variables:
                    env[key] = str(root / "sdk")
            result = subprocess.run(
                ["bash", "scripts/run_cpp_tidy.sh", "--all" if mode == "all" else "--changed-only"],
                cwd=root, env=env, text=True, capture_output=True, check=False,
            )
            calls = log.read_text().splitlines() if log.exists() else []
            return result, calls

    def test_non_cpp_changes_do_not_bootstrap_or_configure(self) -> None:
        for changed in ("README.md", "include/sample.h"):
            with self.subTest(changed=changed):
                result, calls = self.run_tidy(changed, failure=42)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("No C/C++ sources selected", result.stdout)
                self.assertEqual(calls, [])

    def test_cpp_changes_bootstrap_configure_and_analyze(self) -> None:
        result, calls = self.run_tidy("src/sample.cpp")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["bootstrap", "cmake", "tidy"])
        self.assertIn("clang-tidy file count: 1", result.stdout)

    def test_standalone_analysis_does_not_install_dependencies(self) -> None:
        result, calls = self.run_tidy("src/sample.cpp", install=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["cmake", "tidy"])

    def test_all_mode_still_analyzes_sources(self) -> None:
        result, calls = self.run_tidy("README.md", mode="all")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["bootstrap", "cmake", "tidy"])

    def test_bootstrap_failure_is_not_hidden(self) -> None:
        result, calls = self.run_tidy("src/sample.cpp", failure=42)
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertEqual(calls, ["bootstrap"])

    def test_sdk_host_uses_only_explicit_cohort_and_native_configuration(self) -> None:
        result, calls = self.run_tidy("src/sample.cpp", sdk=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["bootstrap", "cmake", "tidy"])

    def test_configuration_failure_prevents_analysis(self) -> None:
        result, calls = self.run_tidy("src/sample.cpp", configure_failure=43)
        self.assertEqual(result.returncode, 43, result.stderr)
        self.assertEqual(calls, ["bootstrap", "cmake"])


class HygieneWorkflowTests(unittest.TestCase):
    def test_bootstrap_is_opt_in_to_the_tidy_step(self) -> None:
        text = (ROOT / ".github/workflows/vulcan-ci.yml").read_text()
        start = text.index("  code-hygiene:")
        job = text[start:text.index("\n  resolve-test-runner:", start)]
        self.assertNotIn("./build.sh --install-deps-only", job)
        self.assertIn('TIDY_INSTALL_DEPS: "1"', job)


if __name__ == "__main__":
    unittest.main()
