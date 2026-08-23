import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "generate_tutorial_docs.py"
)
SPEC = importlib.util.spec_from_file_location("generate_tutorial_docs", MODULE_PATH)
assert SPEC and SPEC.loader
tutorial_docs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tutorial_docs
SPEC.loader.exec_module(tutorial_docs)


README = """# 001 Run a Model

## Metadata

| Field | Value |
| --- | --- |
| Category | Models & Inference |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Labels | Model |

## Concept

Run a model.

## Learning Process

1. Create the model.

## Run

```bash
./run_a_model
```
"""


def _write_module(root: Path, folder: str, localized: bool = True) -> Path:
    module = root / "tutorials" / folder
    module.mkdir(parents=True)
    (module / "README.md").write_text(README, encoding="utf-8")
    (module / "run_a_model.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )
    if localized:
        target = module / "i18n" / "ja" / "README.md"
        target.parent.mkdir(parents=True)
        target.write_text(
            README.replace("Run a model.", "モデルを実行します。"), encoding="utf-8"
        )
    return module


def _write_locale_support(root: Path, modules: list[Path]) -> None:
    locale_root = root / "tutorials" / "i18n"
    locale_root.mkdir(parents=True, exist_ok=True)
    catalog = dict(tutorial_docs.ENGLISH_UI)
    catalog["title"] = "チュートリアル"
    (locale_root / "ja.json").write_text(
        json.dumps(catalog, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "ja": {
            (module / "README.md")
            .relative_to(root)
            .as_posix(): hashlib.sha256((module / "README.md").read_bytes())
            .hexdigest()
            for module in modules
        }
    }
    (locale_root / "translation-sources.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_generates_source_owned_localized_tutorial(tmp_path: Path, monkeypatch) -> None:
    module = _write_module(tmp_path, "001_run_a_model")
    _write_locale_support(tmp_path, [module])
    canonical = tutorial_docs.parse_module(module, tmp_path)
    monkeypatch.chdir(tmp_path)

    locales = tutorial_docs.generate_localized_tutorials(
        tmp_path, [module], [canonical], "main"
    )

    assert locales == ["ja"]
    output = (
        tmp_path
        / "website/i18n/ja/docusaurus-plugin-content-docs/current"
        / "develop-apps/tutorials/models-inference/tutorial_001_run_a_model.mdx"
    ).read_text(encoding="utf-8")
    assert "モデルを実行します。" in output
    assert "int main() { return 0; }" in output
    assert "/tutorials/run-a-model" in output


def test_generated_tutorial_links_respect_docusaurus_base_url(tmp_path: Path) -> None:
    module_dir = _write_module(tmp_path, "001_run_a_model")
    module = tutorial_docs.parse_module(module_dir, tmp_path)

    index = tutorial_docs.render_index([module], "")
    category = tutorial_docs.render_category_index(
        module.category,
        "models-inference",
        "Model tutorials.",
        [module],
    )

    assert (
        '<BaseUrlLink className="overview-link-card" '
        'href="/tutorials/models-inference/">' in index
    )
    assert (
        '<BaseUrlLink className="tutorial-card-image-title" '
        'href="/tutorials/run-a-model">' in category
    )


def test_main_writes_completed_locale_manifest(tmp_path: Path, monkeypatch) -> None:
    module = _write_module(tmp_path, "001_run_a_model")
    _write_locale_support(tmp_path, [module])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["generate_tutorial_docs.py", "--repo-root", str(tmp_path)]
    )

    assert tutorial_docs.main() == 0
    manifest = json.loads(
        (tmp_path / "website/src/generated/tutorial-locales.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest == {"locales": ["ja"]}


def test_rejects_partially_translated_locale(tmp_path: Path) -> None:
    first = _write_module(tmp_path, "001_run_a_model")
    second = _write_module(tmp_path, "002_run_another_model", localized=False)
    _write_locale_support(tmp_path, [first])
    canonical = [
        tutorial_docs.parse_module(module, tmp_path) for module in (first, second)
    ]

    with pytest.raises(ValueError, match="Tutorial locale ja is incomplete"):
        tutorial_docs.generate_localized_tutorials(
            tmp_path, [first, second], canonical, "main"
        )


def test_removes_stale_generated_locale_when_translations_are_removed(
    tmp_path: Path,
) -> None:
    module = _write_module(tmp_path, "001_run_a_model", localized=False)
    canonical = tutorial_docs.parse_module(module, tmp_path)
    output_root = (
        tmp_path
        / "website/i18n/ja/docusaurus-plugin-content-docs/current"
        / "develop-apps/tutorials"
    )
    generated_category = output_root / "legacy-generated-category"
    generated_category.mkdir(parents=True)
    (output_root / "index.md").write_text("# Stale index\n", encoding="utf-8")
    (generated_category / "tutorial_stale.mdx").write_text(
        "# Stale tutorial\n", encoding="utf-8"
    )
    authored_page = output_root / "before-you-run.md"
    authored_page.write_text("# Authored page\n", encoding="utf-8")

    locales = tutorial_docs.generate_localized_tutorials(
        tmp_path, [module], [canonical], "main"
    )

    assert locales == []
    assert not (output_root / "index.md").exists()
    assert not generated_category.exists()
    assert authored_page.read_text(encoding="utf-8") == "# Authored page\n"


def test_rejects_stale_translation(tmp_path: Path) -> None:
    module = _write_module(tmp_path, "001_run_a_model")
    _write_locale_support(tmp_path, [module])
    canonical = tutorial_docs.parse_module(module, tmp_path)
    (module / "README.md").write_text(README + "\nChanged.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="is stale"):
        tutorial_docs.generate_localized_tutorials(
            tmp_path, [module], [canonical], "main"
        )


def test_rejects_translation_that_changes_protected_structure(tmp_path: Path) -> None:
    source = tmp_path / "README.md"
    localized = tmp_path / "ja.md"
    source.write_text("# Title\n\nSee [guide](/guide) and `Model`.\n", encoding="utf-8")
    localized.write_text(
        "# タイトル\n\n[ガイド](/different) と `Model` を参照。\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="changed link targets"):
        tutorial_docs._validate_localized_readme(source, localized)
