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


class RootIndexFileTests(unittest.TestCase):
    def test_omits_configured_repository_only_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            section = root / "section"
            staging.mkdir()
            section.mkdir()
            source_readme = staging / "README.md"
            source_readme.write_text(
                "# Sentinel\n\n"
                "**Documentation:** [English](docs/README.md) | "
                "[한국어](docs/i18n/ko/README.md)\n\n"
                "Sentinel overview.\n",
                encoding="utf-8",
            )

            wrote_index = MODULE.write_root_index_file(
                {
                    "title": "Sentinel",
                    "root_index_file": "README.md",
                    "root_index_omit_line_prefixes": ["**Documentation:**"],
                },
                staging,
                section,
                [],
            )

            self.assertTrue(wrote_index)
            generated = (section / "index.md").read_text(encoding="utf-8")
            self.assertNotIn("docs/i18n/ko/README.md", generated)
            self.assertIn("Sentinel overview.", generated)
            self.assertIn("docs/i18n/ko/README.md", source_readme.read_text(encoding="utf-8"))

    def test_rejects_invalid_omit_line_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            section = root / "section"
            staging.mkdir()
            section.mkdir()
            (staging / "README.md").write_text("# Sentinel\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "root_index_omit_line_prefixes must be a list of non-empty strings",
            ):
                MODULE.write_root_index_file(
                    {
                        "root_index_file": "README.md",
                        "root_index_omit_line_prefixes": "**Documentation:**",
                    },
                    staging,
                    section,
                    [],
                )


class LocalizedAutodocTests(unittest.TestCase):
    def write_i18n_contract(
        self, staging: Path, source: str, ja_hash: str, ko_hash: str,
        excluded_prefixes=None,
    ):
        (staging / "sima-i18n.config.json").write_text(
            json.dumps(
                {
                    "sourceDir": "docs",
                    "translationDir": "docs/i18n/{locale}",
                    "manifest": "docs/i18n/translation-sources.json",
                    "excludedPrefixes": excluded_prefixes or [],
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

    def test_imports_current_translations_and_replaces_existing_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            build_dir = repo_root / "build"
            staging = build_dir / "autodoc/example"
            english = staging / "docs/index.md"
            english.parent.mkdir(parents=True)
            english.write_text("# English\n", encoding="utf-8")
            fallback = english.parent / "fallback.md"
            fallback.write_text("# English fallback\n", encoding="utf-8")
            asset = english.parent / "images/example.png"
            asset.parent.mkdir()
            asset.write_bytes(b"image fixture")
            digest = hashlib.sha256(english.read_bytes()).hexdigest()
            fallback_digest = hashlib.sha256(fallback.read_bytes()).hexdigest()
            self.write_i18n_contract(staging, "docs/index.md", digest, digest)
            manifest_path = staging / "docs/i18n/translation-sources.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for locale in ("ja", "ko"):
                manifest[locale]["docs/fallback.md"] = fallback_digest
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            for locale, text in (("ja", "# 日本語\n"), ("ko", "# 한국어\n")):
                path = staging / f"docs/i18n/{locale}/index.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                (path.parent / "fallback.md").write_text(
                    f"# {locale} fallback\n", encoding="utf-8"
                )

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
            self.assertIn("localized ja (2)", message)
            ja_page = (
                i18n_root / "ja" / MODULE.DOCUSAURUS_DOCS_TRANSLATION_DIR
                / "tools/example/index.md"
            )
            self.assertIn("# 日本語", ja_page.read_text(encoding="utf-8"))
            ja_section = ja_page.parent
            self.assertIn(
                "# ja fallback",
                (ja_section / "fallback.md").read_text(encoding="utf-8"),
            )
            self.assertEqual((ja_section / "images/example.png").read_bytes(), b"image fixture")
            ko_page = stale_destination / "index.md"
            self.assertIn("# 한국어", ko_page.read_text(encoding="utf-8"))
            self.assertFalse((stale_destination / "old.md").exists())
            self.assertFalse((repo_root / "docs-output/tools/example/i18n").exists())

    def test_stale_translation_fails_instead_of_falling_back_to_english(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            build_dir = repo_root / "build"
            staging = build_dir / "autodoc/example"
            english = staging / "docs/index.md"
            english.parent.mkdir(parents=True)
            english.write_text("# Updated English\n", encoding="utf-8")
            digest = hashlib.sha256(english.read_bytes()).hexdigest()
            self.write_i18n_contract(staging, "docs/index.md", digest, "stale")
            for locale, text in (("ja", "# 日本語\n"), ("ko", "# 한국어\n")):
                path = staging / f"docs/i18n/{locale}/index.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")

            i18n_root = repo_root / "website/i18n"
            stale_destination = (
                i18n_root / "ko" / MODULE.DOCUSAURUS_DOCS_TRANSLATION_DIR
                / "tools/example"
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
                with self.assertRaisesRegex(
                    MODULE.SourceLocalizationError,
                    "ko translation is stale.*sima-i18n check --require-complete",
                ):
                    MODULE.process_source(
                        source,
                        repo_root,
                        build_dir,
                        repo_root / "docs-output",
                        i18n_root,
                        ["ja", "ko"],
                    )

            self.assertFalse(stale_destination.exists())

    def test_missing_translation_fails_instead_of_falling_back_to_english(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            english = staging / "docs/index.md"
            omitted = staging / "docs/omitted.md"
            english.parent.mkdir(parents=True)
            english.write_text("# English\n", encoding="utf-8")
            omitted.write_text("# Must be translated\n", encoding="utf-8")
            digest = hashlib.sha256(english.read_bytes()).hexdigest()
            self.write_i18n_contract(staging, "docs/index.md", digest, digest)
            localized = staging / "docs/i18n/ja/index.md"
            localized.parent.mkdir(parents=True)
            localized.write_text("# 日本語\n", encoding="utf-8")
            config = MODULE.load_source_i18n(
                {"docs_subpath": "docs", "localization": True}, staging,
            )

            failures = MODULE.validate_localized_hashes(
                staging, config, "ja", localized.parent,
            )

            self.assertIn("ja translation is missing: docs/omitted.md", failures)

    def test_excluded_source_prefix_does_not_require_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            english = staging / "docs/index.md"
            generated = staging / "docs/generated/reference.md"
            english.parent.mkdir(parents=True)
            generated.parent.mkdir(parents=True)
            english.write_text("# English\n", encoding="utf-8")
            generated.write_text("# Generated reference\n", encoding="utf-8")
            digest = hashlib.sha256(english.read_bytes()).hexdigest()
            self.write_i18n_contract(
                staging,
                "docs/index.md",
                digest,
                digest,
                excluded_prefixes=["docs/generated/"],
            )
            localized = staging / "docs/i18n/ja/index.md"
            localized.parent.mkdir(parents=True)
            localized.write_text("# 日本語\n", encoding="utf-8")
            config = MODULE.load_source_i18n(
                {"docs_subpath": "docs", "localization": True}, staging,
            )

            failures = MODULE.validate_localized_hashes(
                staging, config, "ja", localized.parent,
            )

            self.assertEqual(failures, [])


class AutodocMainTests(unittest.TestCase):
    def run_main_with_result(self, result):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "autodoc.json"
            manifest.write_text(
                json.dumps({"sources": [{"key": "example"}]}),
                encoding="utf-8",
            )
            argv = [
                "autodoc.py",
                "--conf", str(manifest),
                "--repo-root", str(root),
                "--build-dir", "build",
                "--out-root", str(root / "docs"),
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(MODULE, "process_source") as process_source:
                    if isinstance(result, BaseException):
                        process_source.side_effect = result
                    else:
                        process_source.return_value = result
                    return MODULE.main()

    def test_returns_nonzero_for_localization_contract_failure(self):
        result = MODULE.SourceLocalizationError("translation is stale")

        self.assertEqual(self.run_main_with_result(result), 1)

    def test_preserves_best_effort_behavior_for_source_fetch_failure(self):
        self.assertEqual(self.run_main_with_result((False, "git failed")), 0)

    def test_docs_build_propagates_autodoc_exit_status(self):
        build_script = (ROOT / "build.sh").read_text(encoding="utf-8")
        invocation = build_script.split("python3 tools/autodoc.py", 1)[1].split(
            '  echo "Expanding code tabs..."', 1,
        )[0]

        self.assertNotIn("|| true", invocation)


if __name__ == "__main__":
    unittest.main()
