#!/usr/bin/env python3
"""Contract tests for the required, source-owned Kernel Library portal import."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import kernel_docs
import autodoc


class KernelDocsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def export(self, root, locale="en"):
        files = {}
        for name in kernel_docs.REQUIRED:
            p = root / "kernel-library" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# Kernel Library\n")
            files[p.relative_to(root).as_posix()] = kernel_docs.digest(p)
        manifest = {"schema_version": 1, "locale": locale, "files": files,
                    "source": {"dirty": False, "git_revision": "a" * 40, "tree_sha256": "b" * 64}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_checked_export(self):
        self.export(self.root)
        self.assertEqual(kernel_docs.verify_export(self.root, "en")["locale"], "en")

    def test_modified_missing_extra_and_symlink_files_fail(self):
        for mode in ("modified", "missing", "extra", "symlink"):
            with self.subTest(mode=mode):
                root = self.root / mode
                self.export(root)
                p = root / "kernel-library/index.md"
                if mode == "modified": p.write_text("changed")
                elif mode == "missing": p.unlink()
                elif mode == "extra": (root / "other.md").write_text("extra")
                else:
                    p.unlink()
                    p.symlink_to(root / "kernel-library/a65/index.md")
                with self.assertRaises(ValueError): kernel_docs.verify_export(root, "en")

    def test_wrong_locale_and_unidentified_sources_fail(self):
        manifest = self.export(self.root)
        with self.assertRaises(ValueError): kernel_docs.verify_export(self.root, "ja")
        manifest["source"]["dirty"] = True
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError): kernel_docs.verify_export(self.root, "en")

    def test_warm_mount_preserves_other_chapters_and_does_not_export(self):
        build = self.root / "build"
        for locale in kernel_docs.LOCALES:
            self.export(build / "kernel-docs-cache/key" / locale, locale)
        docs = self.root / "docs"
        docs.mkdir()
        (docs / "existing.md").write_text("untouched")
        with patch.object(kernel_docs, "input_key", return_value="key"), \
             patch.object(kernel_docs.subprocess, "check_output", return_value="c" * 40), \
             patch.object(kernel_docs.subprocess, "run") as run:
            kernel_docs.mount(self.root, build, docs, self.root / "i18n")
            run.assert_not_called()
        self.assertEqual((docs / "existing.md").read_text(), "untouched")
        for locale in kernel_docs.LOCALES[1:]:
            self.assertTrue((self.root / "i18n" / locale / "docusaurus-plugin-content-docs/current/kernel-library/index.md").exists())

    def test_staging_follows_kernels_develop_without_pin_or_fallback(self):
        manifest = json.loads((ROOT / "tools/autodoc.conf.json").read_text())
        source = next(s for s in manifest["sources"] if s["key"] == "kernels")
        self.assertFalse(source.get("githash"))
        self.assertFalse(source.get("staging_githash"))
        with patch.dict("os.environ", {"AUTODOC_SNAP_BRANCH": "main"}):
            self.assertEqual(autodoc.resolve_branch_candidates(source, ROOT), (["develop"], None))
        with patch.dict("os.environ", {"DOCS_DEPLOY_ENV": "stg"}), \
             patch.object(autodoc, "acquire_source", return_value="develop") as acquire, \
             patch.object(autodoc, "source_commit", return_value="d" * 40), \
             patch.object(kernel_docs, "mount", return_value="verified"):
            ok, _ = autodoc.process_source(source, self.root, self.root, self.root, self.root / "i18n")
            self.assertTrue(ok)
            self.assertEqual(acquire.call_args.args[1:3], (["develop"], ""))

    def test_production_does_not_fetch_staging_only_source(self):
        with patch.dict("os.environ", {"DOCS_DEPLOY_ENV": "prod"}), \
             patch.object(autodoc, "acquire_source") as acquire:
            ok, _ = autodoc.process_source({"key": "kernels", "staging_only": True},
                                           self.root, self.root, self.root)
            self.assertTrue(ok)
            acquire.assert_not_called()

    def test_production_removes_previous_staging_mounts(self):
        docs = self.root / "docs"
        i18n = self.root / "i18n"
        self.export(docs)
        (docs / "existing.md").write_text("unchanged")
        for locale in kernel_docs.LOCALES[1:]:
            self.export(i18n / locale / "docusaurus-plugin-content-docs/current", locale)
        (self.root / "kernel-docs-import.json").write_text("{}")
        with patch.dict("os.environ", {"DOCS_DEPLOY_ENV": "prod"}):
            autodoc.process_source({"key": "kernels", "staging_only": True,
                                    "format": "kernel-library-export"},
                                   self.root, self.root, docs, i18n)
        self.assertFalse((docs / "kernel-library").exists())
        self.assertFalse((self.root / "kernel-docs-import.json").exists())
        self.assertEqual((docs / "existing.md").read_text(), "unchanged")
        for locale in kernel_docs.LOCALES[1:]:
            self.assertFalse((i18n / locale / "docusaurus-plugin-content-docs/current/kernel-library").exists())

    def test_required_source_failure_is_fatal(self):
        conf = self.root / "sources.json"
        conf.write_text(json.dumps({"sources": [{"key": "kernels", "required": True}]}))
        with patch.object(autodoc, "process_source", return_value=(False, "missing export")), \
             patch.object(sys, "argv", ["autodoc", "--conf", str(conf), "--repo-root", str(self.root),
                                        "--build-dir", "build", "--out-root", "docs"]):
            self.assertEqual(autodoc.main(), 1)


if __name__ == "__main__":
    unittest.main()
