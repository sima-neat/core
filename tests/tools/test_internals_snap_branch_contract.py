#!/usr/bin/env python3
"""Tests for fail-closed matching-branch Internals artifact resolution."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "build.sh"


def run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script, "bash", str(BUILD_SCRIPT)],
        check=False,
        text=True,
        capture_output=True,
        cwd=ROOT,
    )


class InternalsSnapBranchContractTest(unittest.TestCase):
    def test_matching_core_branch_resolves_matching_internals_snap(self) -> None:
        result = run_bash(
            r"""
source "$1"
GITHUB_HEAD_REF=feature/b4586-dmabuf-zero-copy
NEAT_DEPS_MANIFEST=deps/manifest.json
resolve_neat_internals_ref
printf 'REQUESTED=%s\n' "${NEAT_INTERNALS_REQUESTED_REF}"
printf 'REQUIRED=%s\n' "${NEAT_INTERNALS_REQUIRED_BRANCH}"
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("REQUESTED=feature/b4586-dmabuf-zero-copy:latest", result.stdout)
        self.assertIn("REQUIRED=feature/b4586-dmabuf-zero-copy", result.stdout)

    def test_mismatched_core_branch_is_rejected(self) -> None:
        result = run_bash(
            r"""
source "$1"
GITHUB_HEAD_REF=develop
NEAT_DEPS_MANIFEST=deps/manifest.json
resolve_neat_internals_ref
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no develop fallback is permitted", result.stderr)

    def test_cross_branch_resolver_response_is_rejected(self) -> None:
        result = run_bash(
            r"""
source "$1"
NEAT_INTERNALS_REQUIRED_BRANCH=feature/b4586-dmabuf-zero-copy
validate_neat_internals_resolved_ref develop:0123456789ab
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing a cross-branch artifact", result.stderr)

    def test_resolved_sha_is_accepted_as_matching_branch_provenance(self) -> None:
        result = run_bash(
            r"""
source "$1"
NEAT_INTERNALS_REQUIRED_BRANCH=feature/b4586-dmabuf-zero-copy
validate_neat_internals_resolved_ref \
  feature/b4586-dmabuf-zero-copy:0123456789abcdef0123456789abcdef01234567
printf '%s\n' PROVENANCE_OK
"""
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PROVENANCE_OK", result.stdout)

    def test_missing_matching_artifact_never_attempts_develop_fallback(self) -> None:
        result = run_bash(
            r"""
source "$1"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
log="${tmp}/calls.log"
fake="${tmp}/sima-cli"
cat > "${fake}" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "${log}"
case " \$* " in
  *' --help '*) exit 0 ;;
  *' --json '*) exit 42 ;;
esac
exit 0
EOF
chmod +x "${fake}"
SIMA_CLI_BIN="${fake}"
NEAT_INTERNALS_REQUIRED_BRANCH=feature/b4586-dmabuf-zero-copy
NEAT_INTERNALS_SNAP_POLICY=ON
NEAT_INTERNALS_SNAP_TAG_POLICY=OFF
fetch_neat_internals_vulcan_artifacts \
  feature/b4586-dmabuf-zero-copy:latest "${tmp}/out"
"""
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to resolve internals Vulcan artifact", result.stderr)
        self.assertNotIn("retrying develop:latest", result.stderr)


if __name__ == "__main__":
    unittest.main()
