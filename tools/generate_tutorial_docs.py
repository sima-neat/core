#!/usr/bin/env python3
"""Generate docs/tutorials/*.md from tutorials/tutorial_*.cpp sources.

The generator pulls:
- Story / What you learn / Prereqs / Try blocks from top-of-file comments.
- CLI arguments from print_help() std::cout lines.
- Tutorial ordering and labels from tutorials/CMakeLists.txt.
- API links for symbols in "What you'll learn" from docs/reference/cppapi.
"""

from __future__ import annotations

import argparse
import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

REPO_LINK_PREFIX = "https://github.com/manuel-roldan/PipelineSession/blob/main"


@dataclass
class TutorialInfo:
    exe: str
    source: pathlib.Path
    tutorial_id: int
    slug: str
    labels: List[str] = field(default_factory=list)
    story: str = ""
    learn: List[str] = field(default_factory=list)
    prereqs: List[str] = field(default_factory=list)
    try_cmds: List[str] = field(default_factory=list)
    args: List[Tuple[str, str]] = field(default_factory=list)
    has_help: bool = True
    has_print_gst: bool = False
    main_function: str = ""
    main_function_start_line: int = 0
    learn_linked: List[str] = field(default_factory=list)


_SECTION_KEYS = {
    "story": "story",
    "what you learn": "learn",
    "prereqs": "prereqs",
    "prerequisites": "prereqs",
    "try": "try_cmds",
}


@dataclass
class ApiIndex:
    symbol_to_url: Dict[str, str] = field(default_factory=dict)
    method_to_urls: Dict[str, Set[str]] = field(default_factory=dict)
    qualified_method_to_url: Dict[str, str] = field(default_factory=dict)

    def add_symbol(self, symbol: str, url: str) -> None:
        if not symbol:
            return
        self.symbol_to_url.setdefault(symbol, url)
        if "::" in symbol:
            self.symbol_to_url.setdefault(symbol.split("::")[-1], url)

    def add_method(self, class_symbol: str, method_name: str, url: str) -> None:
        if not class_symbol or not method_name:
            return
        short_class = class_symbol.split("::")[-1]
        keys = {
            f"{class_symbol}::{method_name}",
            f"{short_class}::{method_name}",
        }
        for key in keys:
            self.qualified_method_to_url.setdefault(key, url)
        self.method_to_urls.setdefault(method_name, set()).add(url)

    def unique_method_url(self, method_name: str) -> Optional[str]:
        urls = self.method_to_urls.get(method_name)
        if not urls or len(urls) != 1:
            return None
        return next(iter(urls))


def parse_slug(page_text: str) -> Optional[str]:
    m = re.search(r"^slug:\s*(\S+)\s*$", page_text, flags=re.M)
    return m.group(1).strip() if m else None


def parse_page_symbol(page_text: str) -> Optional[str]:
    m = re.search(r"^#\s+`([^`]+)`\s+(Class|Struct|Namespace|Enum|Typedef|File)\s*$", page_text, flags=re.M)
    return m.group(1).strip() if m else None


def parse_page_qualified_symbol(page_text: str) -> Optional[str]:
    m = re.search(
        r'<div class="doxyDeclaration">\s*(?:class|struct|namespace)\s+([A-Za-z0-9_:]+)\b',
        page_text,
        flags=re.S,
    )
    return m.group(1).strip() if m else None


def parse_class_method_anchors(page_text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for anchor, name in re.findall(r'<a href="#([^"]+)">([~A-Za-z_][A-Za-z0-9_]*)</a>\s*\(', page_text):
        if name in {"operator", "if", "for", "while"}:
            continue
        out.append((name, anchor))
    return out


def build_api_index(reference_api_dir: pathlib.Path) -> ApiIndex:
    idx = ApiIndex()
    if not reference_api_dir.exists():
        return idx

    class_files = sorted((reference_api_dir / "classes").glob("*.md"))
    struct_files = sorted((reference_api_dir / "structs").glob("*.md"))
    namespace_files = sorted((reference_api_dir / "namespaces").glob("*.md"))

    for path in class_files + struct_files + namespace_files:
        text = path.read_text(encoding="utf-8")
        slug = parse_slug(text)
        symbol = parse_page_symbol(text)
        qualified_symbol = parse_page_qualified_symbol(text)
        if not slug or not symbol:
            continue
        idx.add_symbol(symbol, slug)
        if qualified_symbol:
            idx.add_symbol(qualified_symbol, slug)

        if path.parent.name == "classes":
            for method_name, anchor in parse_class_method_anchors(text):
                idx.add_method(symbol, method_name, f"{slug}#{anchor}")

    return idx


def _link_code(token: str, url: str) -> str:
    return f"[`{token}`]({url})"


def linkify_learn_item(text: str, api: ApiIndex) -> str:
    if not text:
        return text

    # Gather candidate replacements and apply longest non-overlapping first.
    candidates: List[Tuple[int, int, str]] = []

    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_:]*)::([A-Za-z_~][A-Za-z0-9_]*)\(\)", text):
        token = m.group(0)
        class_part = m.group(1)
        method = m.group(2)
        key_exact = f"{class_part}::{method}"
        key_short = f"{class_part.split('::')[-1]}::{method}"
        url = api.qualified_method_to_url.get(key_exact) or api.qualified_method_to_url.get(key_short)
        if url:
            candidates.append((m.start(), m.end(), _link_code(token, url)))

    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\(\)", text):
        token = m.group(0)
        name = m.group(1)
        url = api.unique_method_url(name)
        if url:
            candidates.append((m.start(), m.end(), _link_code(token, url)))

    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_:]*::[A-Za-z_][A-Za-z0-9_]*)\b", text):
        token = m.group(1)
        url = api.symbol_to_url.get(token)
        if url:
            candidates.append((m.start(), m.end(), _link_code(token, url)))

    for m in re.finditer(r"\b([A-Z][A-Za-z0-9_]+)\b", text):
        token = m.group(1)
        # Avoid linking partial identifiers inside qualified names.
        left = text[m.start() - 2 : m.start()] if m.start() >= 2 else ""
        right = text[m.end() : m.end() + 2]
        if left == "::" or right == "::":
            continue
        url = api.symbol_to_url.get(token)
        if url:
            candidates.append((m.start(), m.end(), _link_code(token, url)))

    if not candidates:
        return text

    candidates.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    chosen: List[Tuple[int, int, str]] = []
    occupied: List[Tuple[int, int]] = []
    for start, end, repl in candidates:
        overlap = False
        for os, oe in occupied:
            if not (end <= os or start >= oe):
                overlap = True
                break
        if overlap:
            continue
        chosen.append((start, end, repl))
        occupied.append((start, end))

    chosen.sort(key=lambda x: x[0])
    out: List[str] = []
    cursor = 0
    for start, end, repl in chosen:
        if start < cursor:
            continue
        out.append(text[cursor:start])
        out.append(repl)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def parse_cmake_tutorials(cmake_text: str) -> List[Tuple[str, List[str]]]:
    entries: List[Tuple[str, List[str]]] = []
    lines = cmake_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("add_tutorial("):
            block: List[str] = [line]
            i += 1
            while i < len(lines):
                block.append(lines[i].rstrip())
                if lines[i].strip() == ")":
                    break
                i += 1
            joined = "\n".join(block)
            m_name = re.search(r"add_tutorial\(\s*(tutorial_\d{4}_[A-Za-z0-9_]+)", joined)
            if not m_name:
                i += 1
                continue
            exe = m_name.group(1)
            labels: List[str] = []
            m_labels = re.search(r"LABELS\s+([^\n\)]+)", joined)
            if m_labels:
                labels = [tok for tok in m_labels.group(1).split() if tok and tok != ")"]
            entries.append((exe, labels))
        i += 1
    return entries


def read_header_comment(source_text: str) -> List[str]:
    lines = source_text.splitlines()
    out: List[str] = []

    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return out

    first = lines[i].lstrip()
    if first.startswith("//"):
        while i < len(lines):
            raw = lines[i]
            stripped = raw.lstrip()
            if stripped.startswith("//"):
                out.append(stripped[2:].lstrip())
                i += 1
                continue
            if not stripped:
                out.append("")
                i += 1
                continue
            break
        return out

    if first.startswith("/*"):
        while i < len(lines):
            raw = lines[i].strip()
            raw = re.sub(r"^/\*+", "", raw)
            raw = re.sub(r"\*+/\s*$", "", raw)
            raw = re.sub(r"^\*\s?", "", raw)
            out.append(raw)
            if "*/" in lines[i]:
                break
            i += 1
        return out

    return out


def parse_header_sections(comment_lines: List[str]) -> Dict[str, object]:
    result: Dict[str, object] = {
        "story": "",
        "learn": [],
        "prereqs": [],
        "try_cmds": [],
    }
    if not comment_lines:
        return result

    current: str | None = None
    for raw in comment_lines:
        line = raw.strip()
        if not line:
            continue

        m = re.match(r"^([A-Za-z ]+):\s*(.*)$", line)
        if m:
            key = m.group(1).strip().lower()
            val = m.group(2).strip()
            current = _SECTION_KEYS.get(key)
            if current == "story" and val:
                result["story"] = val
            elif current in ("learn", "prereqs", "try_cmds") and val:
                if val.startswith("- "):
                    val = val[2:].strip()
                result[current].append(val)
            continue

        if line.startswith("-") and current in ("learn", "prereqs"):
            result[current].append(line.lstrip("-").strip())
            continue

        if current == "try_cmds":
            if line.startswith("./") or line.startswith("build/"):
                result["try_cmds"].append(line)
            continue

    if not result["story"]:
        for line in comment_lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith("tutorial_"):
                continue
            if s.lower().startswith("story:"):
                continue
            if s.lower().startswith("what you learn"):
                continue
            result["story"] = s
            break

    return result


def parse_print_help_args(source_text: str) -> List[Tuple[str, str]]:
    args: List[Tuple[str, str]] = []
    m = re.search(r"void\s+print_help\s*\([^)]*\)\s*\{(.*?)\n\}", source_text, flags=re.S)
    if not m:
        return args
    body = m.group(1)

    literals = re.findall(r"std::cout\s*<<\s*\"([^\"]*)\"", body)
    for lit in literals:
        text = lit.replace("\\n", "\n")
        for line in text.split("\n"):
            entry = line.strip()
            if not entry.startswith("--"):
                continue
            parts = re.split(r"\s{2,}", entry, maxsplit=1)
            flag = parts[0].strip()
            desc = parts[1].strip() if len(parts) > 1 else ""
            args.append((flag, desc))
    dedup: List[Tuple[str, str]] = []
    seen = set()
    for flag, desc in args:
        if flag in seen:
            continue
        seen.add(flag)
        dedup.append((flag, desc))
    return dedup


def extract_main_function(source_text: str) -> Tuple[str, int]:
    match = re.search(r"\bint\s+main\s*\([^)]*\)\s*\{", source_text)
    if not match:
        return "", 0

    start = match.start()
    start_line = source_text.count("\n", 0, start) + 1
    open_brace = source_text.find("{", match.start())
    if open_brace < 0:
        return "", 0

    depth = 0
    i = open_brace
    in_line_comment = False
    in_block_comment = False
    in_string = False
    quote = ""
    escaped = False

    while i < len(source_text):
        ch = source_text[i]
        nxt = source_text[i + 1] if i + 1 < len(source_text) else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if escaped:
                escaped = False
                i += 1
                continue
            if ch == "\\":
                escaped = True
                i += 1
                continue
            if ch == quote:
                in_string = False
                quote = ""
                i += 1
                continue
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch in ("\"", "'"):
            in_string = True
            quote = ch
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source_text[start : i + 1].strip(), start_line
        i += 1

    return "", 0


def title_from_slug(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.split("_"))


def description_from_story(story: str, fallback_title: str) -> str:
    if story:
        story = story.strip()
        if len(story) > 95:
            story = story[:92].rstrip() + "..."
        return story
    return f"{fallback_title} tutorial."


def md_for_tutorial(t: TutorialInfo) -> str:
    title = title_from_slug(t.slug)
    description = description_from_story(t.story, title)
    sidebar_position = t.tutorial_id + 1

    lines: List[str] = []
    lines.append("---")
    lines.append(f"title: {title}")
    lines.append(f"description: \"{description}\"")
    lines.append(f"sidebar_position: {sidebar_position}")
    lines.append("---")
    lines.append("")
    lines.append("<!-- AUTO-GENERATED by tools/generate_tutorial_docs.py. -->")
    lines.append(f"# Tutorial {t.tutorial_id:04d} {title}")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(t.story if t.story else "see tutorial source for the implementation details.")
    lines.append("")

    learn_items = t.learn_linked if t.learn_linked else t.learn
    if learn_items:
        lines.append("## What You'll Learn")
        lines.append("")
        for item in learn_items:
            lines.append(f"- {item}")
        lines.append("")

    if t.prereqs:
        lines.append("## Prerequisites")
        lines.append("")
        for item in t.prereqs:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## How To Run")
    lines.append("")
    lines.append("```bash")
    lines.append(f"./build/tutorials/{t.exe}")
    lines.append("```")
    lines.append("")

    if t.try_cmds:
        lines.append("## Examples")
        lines.append("")
        lines.append("```bash")
        for cmd in t.try_cmds:
            lines.append(cmd)
        lines.append("```")
        lines.append("")

    arg_lines: List[Tuple[str, str]] = []
    if t.has_help:
        arg_lines.append(("--help", "Show this help message"))
    if t.has_print_gst:
        arg_lines.append(("--print-gst", "Print the gst-launch string and exit"))
    for flag, desc in t.args:
        if flag in {"--help", "--print-gst"}:
            continue
        arg_lines.append((flag, desc if desc else ""))

    if arg_lines:
        lines.append("## Arguments")
        lines.append("")
        for flag, desc in arg_lines:
            if desc:
                lines.append(f"- `{flag}`: {desc}")
            else:
                lines.append(f"- `{flag}`")
        lines.append("")

    if t.main_function:
        lines.append("## Main Function")
        lines.append("")
        if t.main_function_start_line > 0:
            lines.append(f"_Source starts at `{t.source.name}:{t.main_function_start_line}`._")
            lines.append("")
        lines.append("```cpp")
        lines.append(t.main_function)
        lines.append("```")
        lines.append("")

    lines.append("## Source")
    lines.append(f"- [View on GitHub]({REPO_LINK_PREFIX}/tutorials/{t.source.name})")
    lines.append("")
    return "\n".join(lines)


def generate_index(tutorials: List[TutorialInfo], out_path: pathlib.Path) -> None:
    lines: List[str] = []
    lines.append("---")
    lines.append("title: Tutorials")
    lines.append("description: Runnable, ordered learning path")
    lines.append("sidebar_position: 1")
    lines.append("---")
    lines.append("")
    lines.append("<!-- AUTO-GENERATED by tools/generate_tutorial_docs.py. -->")
    lines.append("")
    lines.append("# Tutorials")
    lines.append("")
    lines.append("This is the canonical tutorial index for SiMa NEAT. Each tutorial is a")
    lines.append("small, runnable C++ program that tells a focused story about one part of the")
    lines.append("framework. Tutorials are designed to be read in order.")
    lines.append("")
    lines.append("## Build and run")
    lines.append("")
    lines.append("Build everything:")
    lines.append("")
    lines.append("```")
    lines.append("cmake -S . -B build")
    lines.append("cmake --build build")
    lines.append("```")
    lines.append("")
    lines.append("Run a single tutorial:")
    lines.append("")
    lines.append("```")
    lines.append("./build/tutorials/tutorial_0001_quickstart_image_to_tensor")
    lines.append("```")
    lines.append("")
    lines.append("Run the fast tutorial tier in CI or locally:")
    lines.append("")
    lines.append("```")
    lines.append("ctest --test-dir build -L tutorial -L fast")
    lines.append("```")
    lines.append("")
    lines.append("## Common flags")
    lines.append("")
    lines.append("Most tutorials support:")
    lines.append("")
    lines.append("- `--help` show help")
    lines.append("- `--print-gst` print the gst-launch string and exit (when implemented by that tutorial)")
    lines.append("")
    lines.append("## Tiering and prerequisites")
    lines.append("")
    lines.append("- **fast**: no external assets or plugins (always on in CI)")
    lines.append("- **asset**: uses local assets in `examples/media/*` or `tests/assets/*`")
    lines.append("- **mpk**: requires MPK files or downloads")
    lines.append("- **rtsp**: requires RTSP endpoints or local RTSP server")
    lines.append("- **pcie**: requires SoC/host PCIe setup and PCIe GStreamer plugins")
    lines.append("See [Assets and MPK Setup](../how-to/assets_mpk) for asset + MPK setup and download notes.")
    lines.append("")
    lines.append("## Tutorial list (ordered)")
    lines.append("")
    for idx, t in enumerate(tutorials, start=1):
        lines.append(f"{idx}. [{t.exe}]({t.exe})")
    lines.append("")
    lines.append("## Advanced examples (not part of the tutorial sequence)")
    lines.append("")
    lines.append("- `examples/model_resnet50.cpp` (advanced) — direct Model usage.")
    lines.append("- `examples/yolov8_multi_rtsp_demo.cpp` (advanced) — multi-stream RTSP YOLOv8.")
    lines.append("- `examples/mpk_run.cpp` (advanced) — manual ModelMPK wiring.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docs/tutorials markdown from tutorial C++ sources.")
    parser.add_argument("--repo-root", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--no-index", action="store_true", help="Do not rewrite docs/tutorials/index.md")
    args = parser.parse_args()

    root = pathlib.Path(args.repo_root).resolve()
    tutorials_dir = root / "tutorials"
    docs_tutorials_dir = root / "docs" / "tutorials"
    reference_api_dir = root / "docs" / "reference" / "api"
    cmake_path = tutorials_dir / "CMakeLists.txt"

    cmake_entries = parse_cmake_tutorials(cmake_path.read_text(encoding="utf-8"))
    api_index = build_api_index(reference_api_dir)
    tutorials: List[TutorialInfo] = []

    for exe, labels in cmake_entries:
        src = tutorials_dir / f"{exe}.cpp"
        if not src.exists():
            continue
        m = re.match(r"tutorial_(\d{4})_(.+)", exe)
        if not m:
            continue
        tid = int(m.group(1))
        slug = m.group(2)

        text = src.read_text(encoding="utf-8")
        comment_lines = read_header_comment(text)
        sections = parse_header_sections(comment_lines)

        main_fn, main_fn_start = extract_main_function(text)

        t = TutorialInfo(
            exe=exe,
            source=src,
            tutorial_id=tid,
            slug=slug,
            labels=labels,
            story=str(sections["story"]),
            learn=list(sections["learn"]),
            prereqs=list(sections["prereqs"]),
            try_cmds=list(sections["try_cmds"]),
            args=parse_print_help_args(text),
            has_help=True,
            has_print_gst=("wants_print_gst" in text),
            main_function=main_fn,
            main_function_start_line=main_fn_start,
        )
        t.learn_linked = [linkify_learn_item(item, api_index) for item in t.learn]

        # Normalize known stale path hints if present in copied source comments.
        t.try_cmds = [cmd.replace("./build/", "./build/tutorials/") if cmd.startswith("./build/tutorial_") else cmd for cmd in t.try_cmds]
        tutorials.append(t)

    tutorials.sort(key=lambda x: x.tutorial_id)

    docs_tutorials_dir.mkdir(parents=True, exist_ok=True)
    for t in tutorials:
        md_path = docs_tutorials_dir / f"{t.exe}.md"
        md_path.write_text(md_for_tutorial(t), encoding="utf-8")

    if not args.no_index:
        generate_index(tutorials, docs_tutorials_dir / "index.md")

    print(f"Generated {len(tutorials)} tutorial docs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
