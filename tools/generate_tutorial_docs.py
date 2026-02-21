#!/usr/bin/env python3
"""Generate tutorial docs from tutorial module folders.

Source of truth:
- tutorials/00x_*/README.md (metadata + concept/process)
- tutorials/00x_*/*.cpp
- tutorials/00x_*/*.py

Outputs:
- docs/tutorials/tutorial_v2_<folder>.mdx
- docs/tutorials/index.md
"""

from __future__ import annotations

import argparse
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, List

REPO_LINK_PREFIX = "https://github.com/sima-neat/core/blob/main"


@dataclass
class TutorialModule:
    folder: str
    number: int
    slug: str
    title: str
    difficulty: str
    estimated_read_time: str
    labels: List[str]
    concept: str
    process_steps: List[str]
    run_commands: List[str]
    cpp_rel: str
    py_rel: str

    @property
    def doc_id(self) -> str:
        return f"tutorial_v2_{self.folder}"

    @property
    def doc_slug(self) -> str:
        return f"/tutorials/v2/{self.number:03d}-{self.slug.replace('_', '-')}"


def _extract_section(text: str, heading: str) -> str:
    pattern = rf"^##\s+{re.escape(heading)}\s*$"
    m = re.search(pattern, text, flags=re.M)
    if not m:
        return ""
    start = m.end()
    tail = text[start:]
    next_heading = re.search(r"^##\s+", tail, flags=re.M)
    if next_heading:
        tail = tail[: next_heading.start()]
    return tail.strip()


def _parse_metadata_table(section_text: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in section_text.splitlines():
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) != 2:
            continue
        key, value = cells
        if key.lower() in {"field", "---"}:
            continue
        values[key.lower()] = value
    return values


def _parse_numbered(section_text: str) -> List[str]:
    out: List[str] = []
    for line in section_text.splitlines():
        s = line.strip()
        m = re.match(r"^\d+\.\s+(.*)$", s)
        if m:
            out.append(m.group(1).strip())
    return out


def _parse_run_commands(section_text: str) -> List[str]:
    m = re.search(r"```bash\n(.*?)\n```", section_text, flags=re.S)
    if not m:
        return []
    return [ln.rstrip() for ln in m.group(1).splitlines() if ln.strip()]


def _description(module: TutorialModule) -> str:
    text = module.concept.strip() if module.concept else module.title
    if len(text) > 120:
        return text[:117].rstrip() + "..."
    return text


def parse_module(module_dir: pathlib.Path, repo_root: pathlib.Path) -> TutorialModule:
    name = module_dir.name
    m = re.match(r"^(\d{3})_(.+)$", name)
    if not m:
        raise ValueError(f"Invalid module folder name: {name}")

    number = int(m.group(1))
    slug = m.group(2)

    readme_path = module_dir / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"Missing README: {readme_path}")

    cpp = next(module_dir.glob("*.cpp"), None)
    py = next(module_dir.glob("*.py"), None)
    if not cpp or not py:
        raise FileNotFoundError(f"Missing C++/Python source pair in: {module_dir}")

    text = readme_path.read_text(encoding="utf-8")

    title_match = re.search(r"^#\s+(.+)$", text, flags=re.M)
    title = title_match.group(1).strip() if title_match else f"{number:03d} {slug.replace('_', ' ').title()}"

    meta_section = _extract_section(text, "Metadata")
    concept_section = _extract_section(text, "Concept")
    process_section = _extract_section(text, "Learning Process")
    run_section = _extract_section(text, "Run")

    meta = _parse_metadata_table(meta_section)
    difficulty = meta.get("difficulty", "Intermediate")
    estimated = meta.get("estimated read time", "10-15 minutes")
    labels_raw = meta.get("labels", "")
    labels = [x.strip() for x in labels_raw.split(",") if x.strip()]

    process_steps = _parse_numbered(process_section)
    run_commands = _parse_run_commands(run_section)

    cpp_rel = cpp.resolve().relative_to(repo_root.resolve()).as_posix()
    py_rel = py.resolve().relative_to(repo_root.resolve()).as_posix()

    return TutorialModule(
        folder=name,
        number=number,
        slug=slug,
        title=title,
        difficulty=difficulty,
        estimated_read_time=estimated,
        labels=labels,
        concept=concept_section,
        process_steps=process_steps,
        run_commands=run_commands,
        cpp_rel=cpp_rel,
        py_rel=py_rel,
    )


def render_tutorial_doc(module: TutorialModule, sidebar_position: int) -> str:
    cpp_code = pathlib.Path(module.cpp_rel).read_text(encoding="utf-8").rstrip()
    py_code = pathlib.Path(module.py_rel).read_text(encoding="utf-8").rstrip()

    label_text = ", ".join(f"`{l}`" for l in module.labels) if module.labels else "-"

    lines: List[str] = [
        "---",
        f"id: {module.doc_id}",
        f"title: {module.title}",
        f'description: "{_description(module)}"',
        f"sidebar_position: {sidebar_position}",
        f"slug: {module.doc_slug}",
        "---",
        "",
        "<!-- AUTO-GENERATED by tools/generate_tutorial_docs.py. -->",
        "",
        f"# {module.title}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Difficulty | {module.difficulty} |",
        f"| Estimated Read Time | {module.estimated_read_time} |",
        f"| Labels | {label_text} |",
        "",
        "## Concept",
        module.concept or "Concept summary not provided.",
        "",
        "## Learning Process",
    ]

    if module.process_steps:
        for i, step in enumerate(module.process_steps, 1):
            lines.append(f"{i}. {step}")
    else:
        lines.extend(
            [
                "1. Prepare deterministic inputs and runtime defaults.",
                "2. Execute the chapter runtime flow.",
                "3. Validate behavior from checks and signatures.",
            ]
        )

    lines.extend(["", "## Run", "", "```bash"])
    if module.run_commands:
        lines.extend(module.run_commands)
    else:
        lines.append(f"./tutorial_v2_{module.folder}")
        lines.append(f"python3 {module.py_rel}")
    lines.extend(["```", "", "## Code", "", "<CodeTabs>", '<CodeTab label="C++" lang="cpp">', "", f'```cpp title="{module.cpp_rel}"', cpp_code, "```", "", "</CodeTab>", '<CodeTab label="Python" lang="python">', "", f'```python title="{module.py_rel}"', py_code, "```", "", "</CodeTab>", "</CodeTabs>", "", "## Source", "", f"- C++: [{module.cpp_rel}]({REPO_LINK_PREFIX}/{module.cpp_rel})", f"- Python: [{module.py_rel}]({REPO_LINK_PREFIX}/{module.py_rel})", f"- README: [tutorials/{module.folder}/README.md]({REPO_LINK_PREFIX}/tutorials/{module.folder}/README.md)", ""]) 

    return "\n".join(lines)


def render_index(modules: List[TutorialModule]) -> str:
    lines: List[str] = [
        "---",
        "title: Tutorials",
        "description: Practical tutorials for C++ and Python",
        "sidebar_position: 1",
        "---",
        "",
        "<!-- AUTO-GENERATED by tools/generate_tutorial_docs.py. -->",
        "",
        "# Tutorials",
        "",
        "Use these tutorials in order. Each chapter includes matching C++ and Python implementations.",
        "",
        "| # | Tutorial | Difficulty | Estimated Read Time | Labels |",
        "| --- | --- | --- | --- | --- |",
    ]

    for module in modules:
        labels = ", ".join(module.labels)
        lines.append(
            f"| {module.number:03d} | [{module.title}]({module.doc_id}) | {module.difficulty} | {module.estimated_read_time} | {labels} |"
        )

    lines.append("")
    return "\n".join(lines)


def discover_modules(tutorials_dir: pathlib.Path) -> List[pathlib.Path]:
    items = [
        p
        for p in tutorials_dir.iterdir()
        if p.is_dir() and re.match(r"^\d{3}_.+", p.name)
    ]
    return sorted(items, key=lambda p: p.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate tutorial docs from tutorial module folders.")
    parser.add_argument("--repo-root", default=".", help="Repository root (default: current directory)")
    args = parser.parse_args()

    root = pathlib.Path(args.repo_root).resolve()
    tutorials_dir = root / "tutorials"
    docs_tutorials_dir = root / "docs" / "tutorials"

    module_dirs = discover_modules(tutorials_dir)
    modules = [parse_module(d, root) for d in module_dirs]

    docs_tutorials_dir.mkdir(parents=True, exist_ok=True)

    for idx, module in enumerate(modules, start=2):
        out_path = docs_tutorials_dir / f"{module.doc_id}.mdx"
        out_path.write_text(render_tutorial_doc(module, sidebar_position=idx), encoding="utf-8")

    index_path = docs_tutorials_dir / "index.md"
    index_path.write_text(render_index(modules), encoding="utf-8")

    print(f"Generated {len(modules)} tutorial docs + index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
