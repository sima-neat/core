import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "resolve_compiler_cache.py"
SPEC = importlib.util.spec_from_file_location("resolve_compiler_cache", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ResolveCompilerCacheTests(unittest.TestCase):
    def resolve(self, **overrides):
        values = {
            "base_key_prefix": "core/sccache-v1/arm64/sdk-develop/standard",
            "ref_name": "feature/cache-work",
            "ref_type": "branch",
            "event_name": "push",
            "reader_role_arn": "reader",
            "protected_writer_role_arn": "protected-writer",
            "branch_writer_role_arn": "branch-writer",
            "requested_base_branch": "develop",
            "repo": ROOT,
        }
        values.update(overrides)
        return MODULE.resolve(**values)

    def test_protected_branches_have_separate_writable_namespaces(self):
        develop = self.resolve(ref_name="develop")
        main = self.resolve(ref_name="main")
        self.assertEqual(develop.key_prefix.rsplit("/", 1)[-1], "develop")
        self.assertEqual(main.key_prefix.rsplit("/", 1)[-1], "main")
        self.assertNotEqual(develop.key_prefix, main.key_prefix)
        self.assertEqual((develop.rw_mode, main.rw_mode), ("READ_WRITE", "READ_WRITE"))

    def test_feature_branch_seeds_and_writes_under_selected_base(self):
        result = self.resolve()
        self.assertEqual(
            result.seed_key_prefix,
            "core/sccache-v1/arm64/sdk-develop/standard/develop",
        )
        self.assertEqual(
            result.key_prefix,
            "core/sccache-v1/arm64/sdk-develop/standard/develop/branches/feature%2Fcache-work",
        )
        self.assertEqual(result.rw_mode, "READ_WRITE")

    def test_missing_branch_writer_falls_back_to_selected_base_read_only(self):
        result = self.resolve(branch_writer_role_arn="", requested_base_branch="main")
        self.assertEqual(result.key_prefix.rsplit("/", 1)[-1], "main")
        self.assertEqual(result.seed_key_prefix, "")
        self.assertEqual(result.role_arn, "reader")
        self.assertEqual(result.rw_mode, "READ_ONLY")

    def test_non_direct_context_is_read_only(self):
        result = self.resolve(event_name="pull_request")
        self.assertEqual(result.rw_mode, "READ_ONLY")
        self.assertEqual(result.role_arn, "reader")
        self.assertNotIn("/branches/", result.key_prefix)

    def test_closest_protected_ancestor_wins(self):
        self.assertEqual(
            MODULE.choose_base_branch({"develop": 2, "main": 20}), "develop"
        )
        self.assertEqual(MODULE.choose_base_branch({"develop": 12, "main": 3}), "main")

    def test_equal_ancestry_defaults_to_develop(self):
        self.assertEqual(MODULE.choose_base_branch({"develop": 0, "main": 0}), "develop")

    @mock.patch.object(MODULE, "detect_base_branch", return_value="")
    def test_unrelated_history_disables_remote_cache(self, detect):
        result = self.resolve(requested_base_branch="auto")
        self.assertEqual(result.key_prefix, "")
        self.assertEqual(result.seed_key_prefix, "")
        self.assertEqual(result.role_arn, "")
        self.assertEqual(result.rw_mode, "READ_ONLY")
        self.assertEqual(result.base_branch, "")
        detect.assert_called_once_with(ROOT)

    @mock.patch.object(MODULE, "detect_base_branch", return_value="main")
    def test_auto_base_uses_git_detection(self, detect):
        result = self.resolve(requested_base_branch="auto")
        self.assertEqual(result.base_branch, "main")
        self.assertIn("/main/branches/", result.key_prefix)
        detect.assert_called_once_with(ROOT)


if __name__ == "__main__":
    unittest.main()
