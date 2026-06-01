#!/usr/bin/env python3
"""Generate tutorial docs from tutorial module folders.

Source of truth:
- tutorials/00x_*/README.md (metadata + concept/process)
- tutorials/00x_*/*.cpp
- tutorials/00x_*/*.py

Outputs:
- docs/tutorials/tutorial_<folder>.mdx
- docs/tutorials/index.md
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import pathlib
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List

REPO_LINK_BASE = "https://github.com/sima-neat/core/blob"

# Curated learning-flow order, by tutorial number. Drives both the sidebar
# (TOC) position and the in-section ordering on the tutorials index page.
# Modules whose number is not listed here fall to the end in numeric order.
LEARNING_FLOW_ORDER = [
    1, 2, 3, 4,        # Beginner foundations
    8, 5, 10, 6,       # Core I/O and pre/postprocessing
    7, 9, 11, 12, 17,  # Pipelines, diagnostics, custom graphs, live input
    13, 14, 15, 16,    # Advanced: hybrid graphs, multi-stream, perf, production
]


def _flow_key(number: int) -> tuple:
    """Sort key: explicit flow position first, then numeric fallback."""
    try:
        return (0, LEARNING_FLOW_ORDER.index(number))
    except ValueError:
        return (1, number)


def docs_static_url(path: str) -> str:
    base_url = os.environ.get("DOCS_BASE_URL", "/").strip() or "/"
    if not base_url.startswith("/"):
        base_url = f"/{base_url}"
    if not base_url.endswith("/"):
        base_url = f"{base_url}/"
    return f"{base_url}{path.lstrip('/')}"


def detect_repo_ref(default: str = "main") -> str:
    env_ref = (
        os.environ.get("TUTORIAL_SOURCE_REF")
        or os.environ.get("GITHUB_REF_NAME")
    )
    if env_ref:
        return env_ref.strip()
    try:
        ref = (
            subprocess.check_output(
                ["git", "branch", "--show-current"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
        return ref or default
    except Exception:
        return default


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
    run_section: str
    cpp_rel: str
    py_rel: str

    @property
    def doc_id(self) -> str:
        return f"tutorial_{self.folder}"

    @property
    def doc_slug(self) -> str:
        return f"/tutorials/{self.number:03d}-{self.slug.replace('_', '-')}"

    @property
    def image_url(self) -> str:
        return docs_static_url(f"img/tutorials/cards/{self.difficulty.strip().lower()}.svg")

    @property
    def display_title(self) -> str:
        return re.sub(r"^\d{3}\s+", "", self.title).strip()


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


def _description(module: TutorialModule) -> str:
    text = module.concept.strip() if module.concept else module.title
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 120:
        return text[:117].rstrip() + "..."
    return text


def _yaml_double_quote_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _difficulty_theme(difficulty: str) -> Dict[str, str]:
    key = difficulty.strip().lower()
    if key == "beginner":
        return {
            "bg_a": "#5c7510",
            "bg_b": "#a4cc2d",
            "glow": "#d3ec7d",
            "chip_bg": "#3a4a0a",
            "chip_fg": "#e9f5b8",
        }
    if key == "advanced":
        return {
            "bg_a": "#1d4f86",
            "bg_b": "#5998dd",
            "glow": "#bfdbff",
            "chip_bg": "#15314f",
            "chip_fg": "#d8e7fb",
        }
    return {
        "bg_a": "#15532a",
        "bg_b": "#2a9c4f",
        "glow": "#8be0a3",
        "chip_bg": "#0d3a1c",
        "chip_fg": "#c5ecd1",
    }


def _summary_text(module: TutorialModule) -> str:
    raw = module.concept.strip() if module.concept else ""
    first_para = re.split(r"\n\s*\n", raw, maxsplit=1)[0].strip() if raw else ""
    text = re.sub(r"\s+", " ", first_para).strip()
    if len(text) > 128:
        return text[:125].rstrip() + "..."
    return text or "Learn the key concept and runtime process for this tutorial."


def _line_ranges(nums: List[int]) -> str:
    if not nums:
        return ""
    ranges: List[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return "{" + ",".join(ranges) + "}"


def _render_code_with_core_logic(code: str) -> tuple[str, str]:
    src_lines = code.splitlines()
    out_lines: List[str] = []
    highlighted: List[int] = []
    in_core = False

    start_re = re.compile(r"^\s*(//|#)\s*CORE LOGIC\s*$", flags=re.I)
    end_re = re.compile(r"^\s*(//|#)\s*END CORE LOGIC\s*$", flags=re.I)

    for line in src_lines:
        if start_re.match(line):
            in_core = True
            out_lines.append(line)
            highlighted.append(len(out_lines))
            continue
        if end_re.match(line):
            in_core = False
            continue

        out_lines.append(line)
        if in_core:
            highlighted.append(len(out_lines))

    return "\n".join(out_lines), _line_ranges(highlighted)


def _svg_background(difficulty: str) -> str:
    theme = _difficulty_theme(difficulty)
    digest = hashlib.sha256(difficulty.encode("utf-8")).hexdigest()
    x1 = int(digest[0:2], 16) % 320
    y1 = int(digest[2:4], 16) % 170
    x2 = int(digest[4:6], 16) % 320
    y2 = int(digest[6:8], 16) % 170
    x3 = int(digest[8:10], 16) % 320
    y3 = int(digest[10:12], 16) % 170

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 170" role="img" aria-label="tutorial difficulty background">\n'
        "  <defs>\n"
        f'    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">\n'
        f'      <stop offset="0%" stop-color="{theme["bg_a"]}"/>\n'
        f'      <stop offset="100%" stop-color="{theme["bg_b"]}"/>\n'
        "    </linearGradient>\n"
        f'    <radialGradient id="glow" cx="50%" cy="35%" r="70%">\n'
        f'      <stop offset="0%" stop-color="{theme["glow"]}" stop-opacity="0.28"/>\n'
        f'      <stop offset="100%" stop-color="{theme["glow"]}" stop-opacity="0"/>\n'
        "    </radialGradient>\n"
        "  </defs>\n"
        '  <rect width="320" height="170" rx="16" fill="url(#bg)"/>\n'
        '  <rect width="320" height="170" rx="16" fill="url(#glow)"/>\n'
        f'  <circle cx="{x1}" cy="{y1}" r="58" fill="#ffffff" fill-opacity="0.07"/>\n'
        f'  <circle cx="{x2}" cy="{y2}" r="42" fill="#ffffff" fill-opacity="0.08"/>\n'
        f'  <circle cx="{x3}" cy="{y3}" r="22" fill="#ffffff" fill-opacity="0.14"/>\n'
        '  <path d="M24 128 C 84 84, 184 154, 296 94" stroke="#ffffff" stroke-opacity="0.2" stroke-width="3" fill="none"/>\n'
        "</svg>\n"
    )


def generate_card_images(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("tutorial_*.svg"):
        stale.unlink(missing_ok=True)
    for difficulty in ("beginner", "intermediate", "advanced"):
        svg_path = out_dir / f"{difficulty}.svg"
        svg_path.write_text(_svg_background(difficulty), encoding="utf-8")


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
        run_section=run_section,
        cpp_rel=cpp_rel,
        py_rel=py_rel,
    )


def render_tutorial_doc(module: TutorialModule, sidebar_position: int, repo_ref: str) -> str:
    repo_link_prefix = f"{REPO_LINK_BASE}/{repo_ref}"
    cpp_src = pathlib.Path(module.cpp_rel).read_text(encoding="utf-8").rstrip()
    py_src = pathlib.Path(module.py_rel).read_text(encoding="utf-8").rstrip()
    cpp_code, cpp_hl = _render_code_with_core_logic(cpp_src)
    py_code, py_hl = _render_code_with_core_logic(py_src)

    label_text = ", ".join(f"`{l}`" for l in module.labels) if module.labels else "-"

    lines: List[str] = [
        "---",
        f"id: {module.doc_id}",
        f"title: {module.display_title}",
        f'description: "{_yaml_double_quote_escape(_description(module))}"',
        f"sidebar_position: {sidebar_position}",
        f"slug: {module.doc_slug}",
        "---",
        "",
        "<!-- AUTO-GENERATED by tools/generate_tutorial_docs.py. -->",
        "",
        f"# {module.display_title}",
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

    cpp_fence = f'```cpp title="{module.cpp_rel}"' + (f" {cpp_hl}" if cpp_hl else "")
    py_fence = f'```python title="{module.py_rel}"' + (f" {py_hl}" if py_hl else "")

    # Pass the README's `## Run` section through verbatim so the README is the
    # single source of truth for run instructions — no hardcoded env-var blocks.
    if module.run_section.strip():
        lines.extend(["", "## Run", "", module.run_section])
    lines.extend(
        [
            "",
            "## Code",
            "",
            "<CodeTabs>",
            '<CodeTab label="C++" lang="cpp">',
            "",
            cpp_fence,
            cpp_code,
            "```",
            "",
            "</CodeTab>",
            '<CodeTab label="Python" lang="python">',
            "",
            py_fence,
            py_code,
            "```",
            "",
            "</CodeTab>",
            "</CodeTabs>",
            "",
            "## Source",
            "",
            f"- [C++]({repo_link_prefix}/{module.cpp_rel})",
            f"- [Python]({repo_link_prefix}/{module.py_rel})",
            f"- [README]({repo_link_prefix}/tutorials/{module.folder}/README.md)",
            "",
        ]
    )

    return "\n".join(lines)


def render_index(modules: List[TutorialModule], heading_body: str) -> str:
    groups: Dict[str, List[TutorialModule]] = {
        "Beginner": [],
        "Intermediate": [],
        "Advanced": [],
    }
    for module in modules:
        key = module.difficulty if module.difficulty in groups else "Intermediate"
        groups[key].append(module)
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda m: _flow_key(m.number))

    lines: List[str] = [
        "---",
        "title: Tutorials",
        "description: Practical tutorials for C++ and Python with guided learning paths",
        "sidebar_position: 1",
        "slug: /tutorials",
        "---",
        "",
        "<!-- AUTO-GENERATED by tools/generate_tutorial_docs.py. -->",
        "",
        "# Tutorials",
    ]
    if heading_body:
        lines.extend(["", heading_body.strip(), ""])
    else:
        lines.extend(
            [
                "",
                '<p class="tutorial-grid-intro">Use these tutorials in order. Each card links to a chapter with concept-first guidance and matching C++ and Python implementation.</p>',
                "",
            ]
        )

    for difficulty in ("Beginner", "Intermediate", "Advanced"):
        section = groups[difficulty]
        if not section:
            continue

        lines.extend([f"## {difficulty}", "", '<div class="tutorial-grid">'])

        for module in section:
            title = html.escape(module.display_title)
            summary = html.escape(_summary_text(module))
            diff_class = module.difficulty.strip().lower()
            duration = html.escape(module.estimated_read_time)
            label_tags = "".join(
                f'<span class="tutorial-card-tag">{html.escape(label)}</span>'
                for label in module.labels
            )
            lines.extend(
                [
                    f'  <div class="tutorial-card tutorial-difficulty-{diff_class}">',
                    '    <div class="tutorial-card-image-wrap">',
                    f'      <img class="tutorial-card-image" src="{module.image_url}" alt="{title} image" loading="lazy" />',
                    f'      <a class="tutorial-card-image-title" href="{module.doc_slug}">{title}</a>',
                    f'      <span class="tutorial-card-duration">{duration}</span>',
                    "    </div>",
                    '    <div class="tutorial-card-body">',
                    f'      <p class="tutorial-card-summary">{summary}</p>',
                    f'      <div class="tutorial-card-tags">{label_tags}</div>',
                    "    </div>",
                    "  </div>",
                ]
            )

        lines.extend(["</div>", ""])
    return "\n".join(lines)


def discover_modules(tutorials_dir: pathlib.Path) -> List[pathlib.Path]:
    items = [
        p
        for p in tutorials_dir.iterdir()
        if (
            p.is_dir()
            and re.match(r"^\d{3}_.+", p.name)
            and (
                (p / "README.md").exists()
                or next(p.glob("*.cpp"), None)
                or next(p.glob("*.py"), None)
            )
        )
    ]
    return sorted(items, key=lambda p: p.name)


DIFFICULTY_SUBDIRS = [
    ("Beginner", "beginner", 2),
    ("Intermediate", "intermediate", 3),
    ("Advanced", "advanced", 4),
]


def _difficulty_subdir(difficulty: str) -> str:
    key = (difficulty or "").strip().lower()
    for _, slug, _ in DIFFICULTY_SUBDIRS:
        if slug == key:
            return slug
    return "intermediate"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate tutorial docs from tutorial module folders.")
    parser.add_argument("--repo-root", default=".", help="Repository root (default: current directory)")
    args = parser.parse_args()

    root = pathlib.Path(args.repo_root).resolve()
    tutorials_dir = root / "tutorials"
    docs_tutorials_dir = root / "docs" / "tutorials"

    module_dirs = discover_modules(tutorials_dir)
    modules = [parse_module(d, root) for d in module_dirs]
    modules.sort(key=lambda m: _flow_key(m.number))
    repo_ref = detect_repo_ref()

    docs_tutorials_dir.mkdir(parents=True, exist_ok=True)
    generate_card_images(root / "website" / "static" / "img" / "tutorials" / "cards")
    heading_path = docs_tutorials_dir / "heading.mm"
    heading_body = heading_path.read_text(encoding="utf-8").strip() if heading_path.exists() else ""

    # Create difficulty subdirectories with category metadata so the sidebar
    # nests tutorials under Beginner / Intermediate / Advanced.
    for label, slug, position in DIFFICULTY_SUBDIRS:
        sub = docs_tutorials_dir / slug
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "_category_.json").write_text(
            f'{{\n  "label": "{label}",\n  "position": {position},\n  "collapsed": true\n}}\n',
            encoding="utf-8",
        )

    # Purge stale auto-generated MDX from prior generator runs (flat layout or
    # renamed/removed tutorial folders). Docusaurus indexes everything it sees,
    # so we sweep the parent directory and every difficulty subdir.
    expected_paths = {
        docs_tutorials_dir / _difficulty_subdir(m.difficulty) / f"{m.doc_id}.mdx"
        for m in modules
    }
    for parent in [docs_tutorials_dir, *(docs_tutorials_dir / s for _, s, _ in DIFFICULTY_SUBDIRS)]:
        for stale in parent.glob("tutorial_*.mdx"):
            if stale not in expected_paths:
                stale.unlink()

    # Re-number sidebar_position per difficulty group so each subsection starts at 1.
    per_group_idx: Dict[str, int] = {slug: 0 for _, slug, _ in DIFFICULTY_SUBDIRS}
    for module in modules:
        sub_slug = _difficulty_subdir(module.difficulty)
        per_group_idx[sub_slug] += 1
        out_path = docs_tutorials_dir / sub_slug / f"{module.doc_id}.mdx"
        out_path.write_text(
            render_tutorial_doc(module, sidebar_position=per_group_idx[sub_slug], repo_ref=repo_ref),
            encoding="utf-8",
        )

    index_path = docs_tutorials_dir / "index.md"
    index_path.write_text(render_index(modules, heading_body), encoding="utf-8")

    print(f"Generated {len(modules)} tutorial docs + index (source ref: {repo_ref}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    repo_link_prefix = f"{REPO_LINK_BASE}/{repo_ref}"
