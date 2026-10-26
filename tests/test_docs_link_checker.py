"""Exercise the shell gate's bounded retries and failure diagnostics."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DocsLinkCheckerTests(unittest.TestCase):
    def run_checker(self, failures):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "site").mkdir()
            (root / "curl").write_text("#!/bin/sh\nexit 0\n")
            (root / "npx").write_text(
                "#!/bin/bash\n"
                'if [[ "$2" == serve@* ]]; then\n'
                '  echo "fixture server: GET /missing 404"\n'
                '  exec sleep 60\n'
                "fi\n"
                'echo "checker arguments: $*" >&2\n'
                'count=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)\n'
                'count=$((count + 1))\n'
                'echo "$count" > "$COUNT_FILE"\n'
                'if (( count <= FAILURES )); then\n'
                '  echo "[404] http://localhost:12345/missing" >&2\n'
                '  exit 13\n'
                'fi\n'
            )
            for name in ("curl", "npx"):
                (root / name).chmod(0o755)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}",
                       COUNT_FILE=str(root / "count"), FAILURES=str(failures),
                       DOCS_LINK_SITE_DIR=str(root / "site"),
                       DOCS_LINK_CHECK_PORT="12345", DOCS_LINK_BASE_PATH="/",
                       DOCS_LINK_EXTERNAL_ROUTES="", DOCS_LINK_CHECK_ATTEMPTS="2")
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/ci/check_docs_links.sh")],
                env=env, capture_output=True, text=True, timeout=20,
            )
            return result, int((root / "count").read_text())

    def test_transient_failure_recovers(self):
        result, count = self.run_checker(1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(count, 2)
        self.assertIn("attempt 1/2 exited 13", result.stderr)
        self.assertIn("linkinator@8.1.0", result.stderr)
        self.assertIn("--verbosity debug", result.stderr)
        self.assertNotIn("last 100 local server log lines", result.stderr)

    def test_failure_stays_fatal_and_prints_diagnostics(self):
        result, count = self.run_checker(2)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(count, 2)
        self.assertIn("attempt 2/2 exited 13", result.stderr)
        self.assertIn("[404] http://localhost:12345/missing", result.stderr)
        self.assertIn("fixture server: GET /missing 404", result.stderr)
        self.assertIn("timeout_ms=60000; retries=3", result.stderr)


if __name__ == "__main__":
    unittest.main()
