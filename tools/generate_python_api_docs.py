#!/usr/bin/env python3
"""Generate Python API reference docs organized by module.

Primary docs live under /reference/pythonapi/modules.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODULE_CLASS_RE = re.compile(r'nb::class_<[^>]+>\(m,\s*"(?P<name>[A-Za-z0-9_]+)"\)')
MODULE_DEF_RE = re.compile(r'm\.def_submodule\("(?P<name>[A-Za-z0-9_]+)"')


@dataclass(frozen=True)
class MethodDoc:
    name: str
    doc: str | None


@dataclass(frozen=True)
class PropertyDoc:
    name: str
    doc: str | None


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_module_classes(module_cpp: Path) -> list[str]:
    if not module_cpp.exists():
        return []
    text = read_text(module_cpp)
    names = [m.group("name") for m in MODULE_CLASS_RE.finditer(text)]
    seen = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def extract_submodules(module_cpp: Path) -> list[str]:
    if not module_cpp.exists():
        return []
    text = read_text(module_cpp)
    names = [m.group("name") for m in MODULE_DEF_RE.finditer(text)]
    seen = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def extract_module_docstring(path: Path, fallback: str) -> str:
    if not path.exists():
        return fallback
    text = read_text(path)
    m = re.search(r'"""(.*?)"""', text, re.DOTALL)
    if not m:
        return fallback
    doc = m.group(1).strip()
    return doc if doc else fallback


def render_list(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def extract_class_blocks(module_cpp: Path) -> dict[str, str]:
    if not module_cpp.exists():
        return {}
    text = read_text(module_cpp)
    matches = list(MODULE_CLASS_RE.finditer(text))
    blocks: dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        blocks[match.group("name")] = text[start:end]
    return blocks


def extract_methods(block: str) -> list[MethodDoc]:
    methods: list[MethodDoc] = []
    for match in re.finditer(r"\.def(?:_static)?\((.*?)\)\s*;?", block, re.DOTALL):
        args = match.group(1)
        name_match = re.match(r"\s*\"([^\"]+)\"", args)
        if not name_match:
            continue
        name = name_match.group(1)
        literals = re.findall(r'\"([^\"]+)\"', args)
        doc = None
        if len(literals) >= 2:
            candidate = literals[-1]
            if any(ch.isspace() for ch in candidate) or "." in candidate:
                doc = candidate
        methods.append(MethodDoc(name=name, doc=doc))
    return methods


def extract_class_doc(block: str) -> str | None:
    init_match = re.search(r"\.def\(\s*nb::init<[^>]*>\(\)\s*,\s*\"([^\"]+)\"\s*\)", block)
    if init_match:
        return init_match.group(1)
    return None


def extract_properties(block: str) -> list[PropertyDoc]:
    props: list[PropertyDoc] = []
    for match in re.finditer(r'\.def_(?:rw|prop_ro)\((.*?)\)\s*;?', block, re.DOTALL):
        args = match.group(1)
        name_match = re.match(r'\s*\"([^\"]+)\"', args)
        if not name_match:
            continue
        name = name_match.group(1)
        literals = re.findall(r'\"([^\"]+)\"', args)
        doc = None
        if len(literals) >= 2:
            candidate = literals[-1]
            if any(ch.isspace() for ch in candidate) or "." in candidate:
                doc = candidate
        props.append(PropertyDoc(name=name, doc=doc))
    seen = set()
    ordered: list[PropertyDoc] = []
    for prop in props:
        if prop.name not in seen:
            seen.add(prop.name)
            ordered.append(prop)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    docs_dir = root / "docs"
    out_dir = docs_dir / "reference" / "pythonapi"

    if not docs_dir.exists():
        raise SystemExit(f"Docs directory not found: {docs_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "_category_.json").write_text(
        json.dumps(
            {
                "label": "Python Reference",
                "position": 5,
                "link": {"type": "doc", "id": "reference/pythonapi/index"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_text(
        out_dir / "index.md",
        """---
"""
        "title: Python API Reference\n"
        "description: Python (pyneat) reference\n"
        "slug: /reference/pythonapi\n"
        "---\n\n"
        "Python reference pages are organized by module. The core API lives in `pyneat`, "
        "with helper modules under `pyneat._wrappers` and `pyneat.typing`.\n",
    )

    modules_dir = out_dir / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    (modules_dir / "_category_.json").write_text(
        json.dumps({"label": "Modules", "position": 1}, indent=2) + "\n",
        encoding="utf-8",
    )

    module_cpp = root / "python" / "src" / "module.cpp"
    module_classes = extract_module_classes(module_cpp)
    submodules = extract_submodules(module_cpp)

    core_doc = extract_module_docstring(
        root / "python" / "pyneat" / "__init__.py",
        "SiMa NEAT Python bindings.",
    )
    typing_doc = extract_module_docstring(
        root / "python" / "pyneat" / "typing.py",
        "Typing helpers for pyneat.",
    )

    pyneat_dir = modules_dir / "pyneat"
    pyneat_dir.mkdir(parents=True, exist_ok=True)
    (pyneat_dir / "_category_.json").write_text(
        json.dumps(
            {
                "label": "pyneat",
                "position": 1,
                "link": {"type": "doc", "id": "reference/pythonapi/modules/pyneat/index"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    class_links = [
        f"[{{}}](/reference/pythonapi/modules/pyneat/{{}})".format(c, c)
        for c in module_classes
    ]

    extra_modules = ["pyneat.typing"]

    write_text(
        pyneat_dir / "index.md",
        f"""---
"""
        "title: pyneat\n"
        "slug: /reference/pythonapi/modules/pyneat\n"
        "---\n\n"
        f"{core_doc}\n\n"
        "## Core classes\n"
        f"{render_list(class_links)}\n\n"
        "## Submodules\n"
        f"{render_list([f'pyneat.{name}' for name in submodules] + extra_modules)}\n",
    )

    write_text(
        modules_dir / "pyneat.typing.md",
        f"""---
"""
        "title: pyneat.typing\n"
        "slug: /reference/pythonapi/modules/pyneat.typing\n"
        "---\n\n"
        f"{typing_doc}\n",
    )


    class_blocks = extract_class_blocks(module_cpp)
    for class_name in module_classes:
        block = class_blocks.get(class_name, "")
        class_doc = extract_class_doc(block)
        methods = extract_methods(block)
        properties = extract_properties(block)
        method_lines = []
        for method in methods:
            if method.doc:
                method_lines.append(f"- `{method.name}`: {method.doc}")
            else:
                method_lines.append(f"- `{method.name}`")
        method_section = "\n".join(method_lines) if method_lines else "- (No methods discovered)"
        prop_lines = []
        for prop in properties:
            if prop.doc:
                prop_lines.append(f"- `{prop.name}`: {prop.doc}")
            else:
                prop_lines.append(f"- `{prop.name}`")
        prop_section = "\n".join(prop_lines) if prop_lines else "- (No properties discovered)"
        write_text(
            pyneat_dir / f"{class_name}.md",
            f"""---
"""
            f"title: pyneat.{class_name}\n"
            f"slug: /reference/pythonapi/modules/pyneat/{class_name}\n"
            "---\n\n"
            f"{class_doc or f'Python bindings for `{class_name}`.'}\n\n"
            "## Properties\n"
            f"{prop_section}\n\n"
            "## Methods\n"
            f"{method_section}\n",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
