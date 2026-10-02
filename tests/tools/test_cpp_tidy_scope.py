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
        self, changed: str, *, install: bool = True, mode: str = "changed", failure: int = 0
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            shutil.copyfile(ROOT / "scripts/run_cpp_tidy.sh", root / "scripts/run_cpp_tidy.sh")
            for folder in ("include", "src", "tutorials", "examples", "tests"):
                (root / folder).mkdir()
            (root / "src/sample.cpp").write_text("int sample() { return 0; }\n")
            (root / "README.md").write_text("Documentation\n")
            (root / "include/sample.h").write_text("int sample();\n")
            log = root / "calls"
            bin_dir = root / "bin"
            bin_dir.mkdir()
            commands = {
                "git": 'if [ "$1" = diff ]; then printf "%s\\n" "$TIDY_TEST_CHANGED"; fi',
                "rg": 'case " $* " in *" --files "*) printf "%s\\n" "$PWD/src/sample.cpp";; *) exit 1;; esac',
                "cmake": 'echo cmake >> "$TIDY_TEST_CALLS"',
                "clang-tidy": ":",
                "run-clang-tidy": 'echo tidy >> "$TIDY_TEST_CALLS"',
            }
            for name, body in commands.items():
                target = bin_dir / name
                target.write_text("#!/bin/sh\n" + body + "\n")
                target.chmod(0o755)
            bootstrap = root / "build.sh"
            bootstrap.write_text(
                '#!/bin/sh\necho bootstrap >> "$TIDY_TEST_CALLS"\n'
                + f"exit {failure}\n"
            )
            bootstrap.chmod(0o755)
            env = dict(os.environ)
            env.update(
                PATH=str(bin_dir) + os.pathsep + env["PATH"],
                TIDY_TEST_CALLS=str(log),
                TIDY_TEST_CHANGED=changed,
                TIDY_INSTALL_DEPS="1" if install else "0",
                TIDY_BASE_REF="fixture-base",
                TIDY_JOBS="1",
            )
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


class HygieneWorkflowTests(unittest.TestCase):
    def test_bootstrap_is_opt_in_to_the_tidy_step(self) -> None:
        text = (ROOT / ".github/workflows/vulcan-ci.yml").read_text()
        start = text.index("  code-hygiene:")
        job = text[start:text.index("\n  resolve-test-runner:", start)]
        self.assertNotIn("./build.sh --install-deps-only", job)
        self.assertIn('TIDY_INSTALL_DEPS: "1"', job)


if __name__ == "__main__":
    unittest.main()
