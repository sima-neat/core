#!/usr/bin/env python3
"""Regression tests for localized Software Algolia record generation."""

from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
