#!/usr/bin/env python3
"""Regression tests for localized Software Algolia record generation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


CORE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = CORE_ROOT / "scripts" / "ci" / "sync_algolia_developer_center_index.py"
SPEC = importlib.util.spec_from_file_location("software_algolia_index", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LocalizedRecordTests(unittest.TestCase):
    def test_generates_language_specific_records_and_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs_dir = root / "docs"
            localized_dir = (
                root
                / "i18n"
                / "uk"
                / "docusaurus-plugin-content-docs"
                / "current"
            )
            (docs_dir / "getting-started").mkdir(parents=True)
            (localized_dir / "getting-started").mkdir(parents=True)
            (docs_dir / "getting-started" / "index.md").write_text(
                "# Getting started\nEnglish body.\n",
                encoding="utf-8",
            )
            (localized_dir / "getting-started" / "index.md").write_text(
                "# Початок роботи\nУкраїнський текст.\n",
                encoding="utf-8",
            )

            records, summary = MODULE.generate_records(
                docs_dir,
                "https://build.neat.sima.ai",
                MODULE.DEFAULT_MAX_RECORD_BYTES,
                root / "i18n",
            )

            self.assertEqual(summary["by_language"], {"en": 1, "uk": 1})
            by_language = {record["language"]: record for record in records}
            self.assertEqual(by_language["en"]["route"], "/software/getting-started")
            self.assertEqual(by_language["uk"]["route"], "/software/uk/getting-started")
            self.assertNotEqual(by_language["en"]["objectID"], by_language["uk"]["objectID"])

    def test_localized_record_uses_frontmatter_slug(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs_dir = root / "docs"
            localized_dir = (
                root
                / "i18n"
                / "ja"
                / "docusaurus-plugin-content-docs"
                / "current"
                / "develop-apps/tutorials/models-inference"
            )
            docs_dir.mkdir()
            localized_dir.mkdir(parents=True)
            (localized_dir / "tutorial_001_run_your_first_model.mdx").write_text(
                "---\nslug: /tutorials/run-your-first-model\n---\n\n# 最初のモデル\n本文。\n",
                encoding="utf-8",
            )

            records, _ = MODULE.generate_records(
                docs_dir,
                "https://developer.sima.ai",
                MODULE.DEFAULT_MAX_RECORD_BYTES,
                root / "i18n",
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(
                records[0]["url"],
                "https://developer.sima.ai/software/ja/tutorials/run-your-first-model",
            )
            self.assertEqual(
                records[0]["route"],
                "/software/ja/tutorials/run-your-first-model",
            )


class ReleaseRecordTests(unittest.TestCase):
    def test_resolves_dependency_only_sources(self) -> None:
        snapshot = json.loads((CORE_ROOT / "website/release-data/current.json").read_text())
        minimum = snapshot["dependencies"]["docker"]["minimum"]["value"]
        sources = {"windows.md": "Docker %%dependency.docker.minimum%% or later.", "plain.md": "Plain text."}
        self.assertEqual(MODULE.resolve_release_sources(sources), {
            "windows.md": f"Docker {minimum} or later.", "plain.md": "Plain text.",
        })
        for token in ("%%dependency.missing.minimum%%", "%%dependency.docker.minimum%"):
            with self.subTest(token=token):
                with self.assertRaisesRegex(RuntimeError, r"windows.md:1:.*dependency"):
                    MODULE.resolve_release_sources({"windows.md": token})

    def test_resolves_versions_in_all_locales_before_parsing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            roots = [docs] + [root / "i18n" / locale / MODULE.I18N_DOCS_SUBDIR for locale in ["ko", "ja", "zh-Hant", "uk"]]
            source = (
                "---\nslug: /sdk-%%release.sdk%%\n---\n"
                "# Neat SDK %%release.sdk%%\n"
                "Install `sdk@%%release.sdk_channel%%`; series %%release.sdk_series%%. "
                "Core %%release.core%%; compiler %%release.model_compiler%%. Minimum SDK 2.1.3.0.\n"
            )
            for doc_root in roots:
                doc_root.mkdir(parents=True)
                (doc_root / "index.md").write_text(source, encoding="utf-8")
            records, _ = MODULE.generate_records(docs, "https://developer.sima.ai", MODULE.DEFAULT_MAX_RECORD_BYTES, root / "i18n")
            snapshot = json.loads((CORE_ROOT / "website/release-data/current.json").read_text())
            values = {key: item["value"] for key, item in snapshot["releases"].items()}
            self.assertEqual({record["language"] for record in records}, {"en", "ko", "ja", "zh-Hant", "uk"})
            for record in records:
                self.assertNotIn("%release", json.dumps(record))
                self.assertEqual(record["title"], f"Neat SDK {values['sdk']}")
                self.assertIn(f"sdk@{values['sdk_channel']}", record["content"])
                self.assertIn(f"series {values['sdk_series']}", record["content"])
                self.assertIn(f"Core {values['core']}", record["content"])
                self.assertIn("Minimum SDK 2.1.3.0", record["content"])
                self.assertTrue(record["url"].endswith(f"/sdk-{values['sdk']}"))
            self.assertEqual((docs / "index.md").read_text(encoding="utf-8"), source)

    def test_invalid_tokens_fail_before_fenced_content_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            docs = Path(directory)
            page = docs / "index.md"
            for token in ("%%release.missing%%", "%%release.sdk%"):
                with self.subTest(token=token):
                    page.write_text(f"# SDK\nVisible prose.\n```bash\n{token}\n```\n", encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, r"index.md:4:.*release"):
                        MODULE.generate_records(docs, "https://developer.sima.ai", MODULE.DEFAULT_MAX_RECORD_BYTES)


class RecordTitleTests(unittest.TestCase):
    """Titles follow the page: H1, then frontmatter title, then the file name (#800)."""

    def _titles(self, english: str, localized: dict[str, str]) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs_dir = root / "docs"
            (docs_dir / "guide").mkdir(parents=True)
            (docs_dir / "guide" / "configure-sdk.md").write_text(english, encoding="utf-8")
            for language, text in localized.items():
                target = root / "i18n" / language / "docusaurus-plugin-content-docs" / "current" / "guide"
                target.mkdir(parents=True)
                (target / "configure-sdk.md").write_text(text, encoding="utf-8")
            records, _ = MODULE.generate_records(
                docs_dir,
                "https://build.neat.sima.ai",
                MODULE.DEFAULT_MAX_RECORD_BYTES,
                root / "i18n",
            )
            self.assertTrue(all(record["hierarchy"]["lvl1"] == record["title"] for record in records))
            return {record["language"]: record["title"] for record in records}

    def test_frontmatter_title_is_used_when_there_is_no_h1(self) -> None:
        titles = self._titles(
            "---\ntitle: Configure the SDK\n---\nEnglish body.\n",
            {
                "ja": "---\ntitle: SDK を設定します。\n---\n日本語の本文。\n",
                "uk": "---\ntitle: 'Налаштування SDK'\nsidebar_position: 2\n---\nУкраїнський текст.\n",
            },
        )
        self.assertEqual(titles, {"en": "Configure the SDK", "ja": "SDK を設定します。", "uk": "Налаштування SDK"})

    def test_h1_still_wins_over_frontmatter_title(self) -> None:
        titles = self._titles(
            "---\ntitle: Configure\n---\n# Configure the Palette SDK\nBody.\n",
            {"ko": "---\ntitle: 구성\n---\n# Palette SDK 구성\n본문.\n"},
        )
        self.assertEqual(titles, {"en": "Configure the Palette SDK", "ko": "Palette SDK 구성"})

    def test_file_name_fallback_is_unchanged(self) -> None:
        titles = self._titles("---\nsidebar_position: 1\n---\nBody.\n", {"zh-Hant": "本文。\n"})
        self.assertEqual(titles, {"en": "Configure Sdk", "zh-Hant": "Configure Sdk"})


if __name__ == "__main__":
    unittest.main()
