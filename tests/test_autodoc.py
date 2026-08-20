import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "autodoc.py"
SPEC = importlib.util.spec_from_file_location("autodoc", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PromoteIndexFileTests(unittest.TestCase):
    def test_rewrites_inbound_markdown_and_html_links(self):
        with tempfile.TemporaryDirectory() as directory:
            section = Path(directory)
            (section / "README.md").write_text("# Landing\n", encoding="utf-8")
            nested = section / "panels" / "power.md"
            nested.parent.mkdir()
            nested.write_text(
                "[Overview](../README.md#overview)\n"
                '<a href="../README.md?view=full">HTML overview</a>\n'
                "[Unrelated](../reports.md)\n",
                encoding="utf-8",
            )

            MODULE.promote_index_file({"index_file": "README.md"}, section)

            self.assertFalse((section / "README.md").exists())
            self.assertTrue((section / "index.md").exists())
            rewritten = nested.read_text(encoding="utf-8")
            self.assertIn("[Overview](../index.md#overview)", rewritten)
            self.assertIn('href="../index.md?view=full"', rewritten)
            self.assertIn("[Unrelated](../reports.md)", rewritten)


class LocalizedAutodocTests(unittest.TestCase):
    def write_i18n_contract(
        self, staging: Path, source: str, ja_hash: str, ko_hash: str,
    ):
        (staging / "sima-i18n.config.json").write_text(
            json.dumps(
                {
                    "sourceDir": "docs",
                    "translationDir": "docs/i18n/{locale}",
                    "manifest": "docs/i18n/translation-sources.json",
                    "locales": {"ja": {"code": "ja"}, "ko": {"code": "ko"}},
                }
            ),
            encoding="utf-8",
        )
        manifest = {"ja": {source: ja_hash}, "ko": {source: ko_hash}}
        manifest_path = staging / "docs/i18n/translation-sources.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_maps_nested_docs_subpath_into_each_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            english = staging / "docs/guides/index.md"
            english.parent.mkdir(parents=True)
            english.write_text("# English\n", encoding="utf-8")
            digest = hashlib.sha256(english.read_bytes()).hexdigest()
            self.write_i18n_contract(staging, "docs/guides/index.md", digest, digest)

            config = MODULE.load_source_i18n(
                {"docs_subpath": "docs/guides", "localization": True}, staging,
            )

            self.assertEqual(
                MODULE.localized_docs_path(staging, config, "ja"),
                staging / "docs/i18n/ja/guides",
            )

    def test_imports_current_translation_and_removes_stale_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            build_dir = repo_root / "build"
            staging = build_dir / "autodoc/example"
            english = staging / "docs/index.md"
            english.parent.mkdir(parents=True)
            english.write_text("# English\n", encoding="utf-8")
            (english.parent / "fallback.md").write_text("# English fallback\n", encoding="utf-8")
            asset = english.parent / "images/example.png"
            asset.parent.mkdir()
            asset.write_bytes(b"image fixture")
            digest = hashlib.sha256(english.read_bytes()).hexdigest()
            self.write_i18n_contract(staging, "docs/index.md", digest, "stale")
            for locale, text in (("ja", "# 日本語\n"), ("ko", "# 한국어\n")):
                path = staging / f"docs/i18n/{locale}/index.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")

            i18n_root = repo_root / "website/i18n"
            stale_destination = (
                i18n_root / "ko" / MODULE.DOCUSAURUS_DOCS_TRANSLATION_DIR / "tools/example"
            )
            stale_destination.mkdir(parents=True)
            (stale_destination / "old.md").write_text("old\n", encoding="utf-8")
            source = {
                "key": "example",
                "title": "Example",
                "repo": "unused",
                "branch": "main",
                "docs_subpath": "docs",
                "mount": "tools/example",
                "localization": True,
            }

            with mock.patch.object(MODULE, "acquire_source", return_value="main"):
                ok, message = MODULE.process_source(
                    source,
                    repo_root,
                    build_dir,
                    repo_root / "docs-output",
                    i18n_root,
                    ["ja", "ko"],
                )

            self.assertTrue(ok, message)
            self.assertIn("localized ja (1)", message)
            ja_page = (
                i18n_root / "ja" / MODULE.DOCUSAURUS_DOCS_TRANSLATION_DIR
                / "tools/example/index.md"
            )
            self.assertIn("# 日本語", ja_page.read_text(encoding="utf-8"))
            ja_section = ja_page.parent
            self.assertTrue((ja_section / "fallback.md").is_file())
            self.assertEqual((ja_section / "images/example.png").read_bytes(), b"image fixture")
            self.assertFalse(stale_destination.exists())
            self.assertFalse((repo_root / "docs-output/tools/example/i18n").exists())


if __name__ == "__main__":
    unittest.main()
