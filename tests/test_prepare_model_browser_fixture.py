import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from unittest import mock
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "prepare_model_browser_fixture.py"
SPEC = importlib.util.spec_from_file_location("prepare_model_browser_fixture", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

MPK = b"resnet mpk bytes"


def write_zip(path: Path, members: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def fake_cli(directory: Path, body: str) -> str:
    script = directory / "sima-cli"
    script.write_text("#!/usr/bin/env python3\nimport json, sys\n" + textwrap.dedent(body))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


class PrepareModelBrowserFixtureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name)
        self.out = self.tmp / "out"

    def run_main(self, sima_cli: str) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = MODULE.main([
                "--id", "resnet_50", "--variant", "w8a8", "--output", str(self.out),
                "--sima-cli", sima_cli,
            ])
        return code, stdout.getvalue(), stderr.getvalue()

    def cli_returning(self, zip_members: dict[str, bytes]) -> str:
        zip_path = write_zip(self.tmp / "dl" / "resnet_50_w8a8_default.zip", zip_members)
        info = {"path": str(zip_path), "run_id": "run-1", "sha256": "abc123"}
        return fake_cli(self.tmp, f"print(json.dumps({info!r}))\n")

    def test_selects_the_expected_mpk_and_logs_the_package(self):
        cli = self.cli_returning({
            "licenses/LICENSE": b"license",
            "resnet_50_w8a16_mpk.tar.gz": b"other variant",
            "resnet_50_w8a8_mpk.tar.gz": MPK,
        })
        code, stdout, stderr = self.run_main(cli)
        self.assertEqual(code, 0, stderr)
        mpk = Path(stdout.strip())
        self.assertEqual(mpk, self.out / "resnet_50_w8a8_mpk.tar.gz")
        self.assertEqual(mpk.read_bytes(), MPK)
        self.assertIn("run_id=run-1", stderr)
        self.assertIn("sha256=abc123", stderr)

    def home_reporting_cli(self, report: Path) -> str:
        zip_path = write_zip(self.tmp / "dl" / "resnet_50_w8a8_default.zip",
                             {"resnet_50_w8a8_mpk.tar.gz": MPK})
        return fake_cli(self.tmp, f"""            import os
            home = os.environ["HOME"]
            with open({str(report)!r}, "w") as f:
                json.dump({{"home": home, "entries": os.listdir(home)}}, f)
            print(json.dumps({{"path": {str(zip_path)!r}}}))
            """)

    def test_on_github_actions_the_cli_runs_with_an_empty_home(self):
        report = self.tmp / "home-report.json"
        cli = self.home_reporting_cli(report)
        with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
            code, _, stderr = self.run_main(cli)
        self.assertEqual(code, 0, stderr)
        seen = json.loads(report.read_text())
        self.assertEqual(seen["entries"], [])
        self.assertNotEqual(Path(seen["home"]).resolve(), Path.home().resolve())
        self.assertIn(f"empty HOME={seen['home']}", stderr)

    def test_outside_github_actions_the_cli_keeps_the_user_home(self):
        report = self.tmp / "home-report.json"
        cli = self.home_reporting_cli(report)
        user_home = self.tmp / "user-home"
        user_home.mkdir()
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_ACTIONS"}
        env["HOME"] = str(user_home)
        with mock.patch.dict(os.environ, env, clear=True):
            code, _, stderr = self.run_main(cli)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(Path(json.loads(report.read_text())["home"]), user_home)
        self.assertNotIn("empty HOME", stderr)

    def test_missing_mpk_names_what_the_zip_holds(self):
        cli = self.cli_returning({"licenses/LICENSE": b"x", "resnet_50_w8a16_mpk.tar.gz": b"y"})
        code, stdout, stderr = self.run_main(cli)
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("has no resnet_50_w8a8_mpk.tar.gz; MPKs in it: resnet_50_w8a16_mpk.tar.gz", stderr)

    def test_empty_mpk_is_refused(self):
        cli = self.cli_returning({"resnet_50_w8a8_mpk.tar.gz": b""})
        code, _, stderr = self.run_main(cli)
        self.assertEqual(code, 1)
        self.assertIn("is empty", stderr)

    def test_invalid_zip_is_refused(self):
        bad = self.tmp / "dl" / "bad.zip"
        bad.parent.mkdir(parents=True)
        bad.write_bytes(b"<html>login</html>")
        cli = fake_cli(self.tmp, f"print(json.dumps({{'path': {str(bad)!r}}}))\n")
        code, _, stderr = self.run_main(cli)
        self.assertEqual(code, 1)
        self.assertIn("is not a readable ZIP", stderr)

    def test_cli_failure_reports_its_output(self):
        cli = fake_cli(self.tmp, "print('Error: No such option: --json', file=sys.stderr)\nsys.exit(2)\n")
        code, _, stderr = self.run_main(cli)
        self.assertEqual(code, 1)
        self.assertIn("exited 2", stderr)
        self.assertIn("No such option: --json", stderr)

    def test_non_json_output_is_refused(self):
        cli = fake_cli(self.tmp, "print('Downloaded to somewhere')\n")
        code, _, stderr = self.run_main(cli)
        self.assertEqual(code, 1)
        self.assertIn("did not print JSON", stderr)

    def test_json_without_a_path_is_refused(self):
        cli = fake_cli(self.tmp, "print(json.dumps({'run_id': 'r'}))\n")
        code, _, stderr = self.run_main(cli)
        self.assertEqual(code, 1)
        self.assertIn("gave no ZIP path", stderr)


if __name__ == "__main__":
    unittest.main()
