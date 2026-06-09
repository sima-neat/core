#!/usr/bin/env python3
"""Generate tutorial docs from tutorial module folders.

Source of truth:
- tutorials/00x_*/README.md (metadata + concept/process)
- tutorials/00x_*/*.cpp
- tutorials/00x_*/*.py

Outputs:
- docs/develop-apps/tutorials/<difficulty>/tutorial_<folder>.mdx
- docs/develop-apps/tutorials/index.md
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

REPO_LINK_BASE = "https://github.com/sima-neat/core/blob"

# Curated learning-flow order, by tutorial number. Drives both the sidebar
# (TOC) position and the in-section ordering on the tutorials index page.
# Modules whose number is not listed here fall to the end in numeric order.
LEARNING_FLOW_ORDER = [
    1, 2, 3, 4, 5,          # Beginner foundations
    9, 6, 11, 7,            # Core I/O and pre/postprocessing
    8, 10, 12, 13, 18,      # Pipelines, diagnostics, custom graphs, live input
    14, 15, 16, 17,         # Advanced: hybrid graphs, multi-stream, perf, production
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
class WalkStep:
    """One narrative step in a tutorial Walkthrough.

    `name` is the slug bound by `{#step-<name>}` in the README and by
    `STEP <name>` markers in the sources. A snippet is "" when the matching
    source segment is absent for that language. `cpp_prose`/`py_prose` carry
    language-specific explanation (authored with `**C++:**` / `**Python:**`
    markers) rendered inside the matching language tab so the procedure toggles
    with the code; `shared_prose` applies to both.
    """

    name: str
    heading: str
    shared_prose: str
    cpp_prose: str = ""
    py_prose: str = ""
    cpp_snippet: str = ""
    py_snippet: str = ""


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
    in_practice: str
    cpp_rel: str
    py_rel: str
    walkthrough_lead: str = ""
    walkthrough_steps: List[WalkStep] = field(default_factory=list)

    @property
    def has_walkthrough(self) -> bool:
        return bool(self.walkthrough_steps)

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
    def flow_image_url(self) -> str:
        return docs_static_url(f"img/tutorials/flow/{self.folder}.svg")

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
    text = module.concept.strip() or module.walkthrough_lead.strip() or module.title
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
    raw = module.concept.strip() or module.walkthrough_lead.strip()
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
        # Walkthrough STEP markers are stripped from the full-file listing so
        # the collapsed "Full source" block stays clean. (No-op for tutorials
        # that don't use them.) Highlighting in the full listing still comes
        # from CORE LOGIC — per-step snippets are intentionally un-highlighted.
        if _STEP_START_RE.match(line) or _STEP_END_RE.match(line):
            continue

        out_lines.append(line)
        if in_core:
            highlighted.append(len(out_lines))

    return "\n".join(out_lines), _line_ranges(highlighted)


_STEP_START_RE = re.compile(r"^\s*(?://|#)\s*STEP\s+([a-z0-9][a-z0-9_-]*)\s*$", flags=re.I)
_STEP_END_RE = re.compile(r"^\s*(?://|#)\s*END STEP(?:\s+[a-z0-9_-]+)?\s*$", flags=re.I)
_STEP_ANCHOR_RE = re.compile(r"\{#step-([a-z0-9][a-z0-9_-]*)\}")
_LANG_PROSE_RE = re.compile(r"(?m)^\*\*(C\+\+|Python):\*\*[ \t]*")


def _extract_named_segments(code: str, source_label: str) -> Dict[str, str]:
    """Extract `STEP <name>`/`END STEP` regions as dedented snippets.

    Returns ``{name: snippet}``. Nested STEP regions are supported (a stack).
    Marker lines are stripped from snippets. Snippets are intentionally
    un-highlighted: highlighting is reserved for the full-source listing (driven
    by CORE LOGIC), since a per-step snippet is already the focused excerpt.
    Raises ValueError on duplicate names, unbalanced markers, or stray END
    markers — all of which name ``source_label``.
    """
    segments: Dict[str, str] = {}
    stack: List[Dict] = []  # each frame: {"name", "body"}

    for lineno, line in enumerate(code.splitlines(), 1):
        m = _STEP_START_RE.match(line)
        if m:
            name = m.group(1)
            if name in segments or any(s["name"] == name for s in stack):
                raise ValueError(
                    f"{source_label}: duplicate STEP name '{name}' (line {lineno})"
                )
            stack.append({"name": name, "body": []})
            continue
        if _STEP_END_RE.match(line):
            if not stack:
                raise ValueError(f"{source_label}: stray 'END STEP' (line {lineno})")
            done = stack.pop()
            segments[done["name"]] = textwrap.dedent("\n".join(done["body"])).rstrip("\n")
            continue

        # Body line: record it in every currently-open step.
        for frame in stack:
            frame["body"].append(line)

    if stack:
        unclosed = ", ".join(s["name"] for s in stack)
        raise ValueError(f"{source_label}: unclosed STEP region(s): {unclosed}")
    return segments


def _split_step_prose(body: str) -> Tuple[str, str, str]:
    """Split a walkthrough step body into (shared, cpp, py) prose.

    Prose before the first ``**C++:**`` / ``**Python:**`` marker is shared; text
    after each marker (until the next marker or end) is that language's prose.
    """
    parts = _LANG_PROSE_RE.split(body)
    shared = parts[0].strip()
    cpp = py = ""
    for i in range(1, len(parts), 2):
        lang = parts[i]
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if lang == "C++":
            cpp = text
        else:
            py = text
    return shared, cpp, py


def _parse_walkthrough(readme_text: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    """Parse the README ``## Walkthrough`` section.

    Returns ``(lead, [(name, heading, body), ...])``. ``lead`` is the prose
    between the heading and the first ``###`` subsection. Each subsection's
    ``{#step-<name>}`` anchor binds it to a source segment; the anchor token is
    stripped from the returned heading. ``body`` is the raw subsection prose
    (later split into shared/per-language parts). A subsection without an anchor
    yields an empty ``name`` (flagged downstream).
    """
    section = _extract_section(readme_text, "Walkthrough")
    if not section:
        return "", []

    parts = re.split(r"^###\s+(.*)$", section, flags=re.M)
    # parts[0] is the lead; thereafter pairs of (heading_line, body).
    lead = parts[0].strip()
    steps: List[Tuple[str, str, str]] = []
    for i in range(1, len(parts), 2):
        heading_line = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        anchor = _STEP_ANCHOR_RE.search(heading_line)
        name = anchor.group(1) if anchor else ""
        heading = _STEP_ANCHOR_RE.sub("", heading_line).strip()
        steps.append((name, heading, body))
    return lead, steps


def _build_walk_steps(
    readme_text: str, cpp_code: str, py_code: str, folder: str
) -> Tuple[str, List[WalkStep]]:
    """Pair README walkthrough prose with source segments (prose order wins)."""
    lead, prose_steps = _parse_walkthrough(readme_text)
    if not prose_steps:
        return "", []

    cpp_segs = _extract_named_segments(cpp_code, f"{folder}/*.cpp")
    py_segs = _extract_named_segments(py_code, f"{folder}/*.py")

    steps: List[WalkStep] = []
    referenced: set = set()
    for name, heading, body in prose_steps:
        shared, cpp_prose, py_prose = _split_step_prose(body)
        if not name:
            print(
                f"WARNING: {folder}: walkthrough step '{heading}' has no "
                f"{{#step-<name>}} anchor; rendering prose without code.",
                file=sys.stderr,
            )
            steps.append(
                WalkStep(
                    name="",
                    heading=heading,
                    shared_prose=shared,
                    cpp_prose=cpp_prose,
                    py_prose=py_prose,
                )
            )
            continue
        referenced.add(name)
        cpp_snip = cpp_segs.get(name, "")
        py_snip = py_segs.get(name, "")
        if not cpp_snip and not py_snip:
            print(
                f"WARNING: {folder}: walkthrough step '{name}' has no matching "
                f"STEP marker in either source; rendering prose without code.",
                file=sys.stderr,
            )
        steps.append(
            WalkStep(
                name=name,
                heading=heading,
                shared_prose=shared,
                cpp_prose=cpp_prose,
                py_prose=py_prose,
                cpp_snippet=cpp_snip,
                py_snippet=py_snip,
            )
        )

    for orphan in sorted((set(cpp_segs) | set(py_segs)) - referenced):
        print(
            f"WARNING: {folder}: source STEP '{orphan}' is not referenced by any "
            f"walkthrough step.",
            file=sys.stderr,
        )
    return lead, steps


def _code_fence(lang: str, title: str, highlight: str) -> str:
    """Opening fence for a titled code block, with optional highlight metastring."""
    fence = f'```{lang} title="{title}"'
    return f"{fence} {highlight}" if highlight else fence


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


def _first_sentence(text: str) -> str:
    """The first complete sentence of `text` (so animation subtitles read as a
    whole sentence rather than a mid-clause truncation)."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    m = re.match(r"(.*?[.!?])(?:\s|$)", t)
    return (m.group(1) if m else t).strip()


def _flow_node_label(slug: str, limit: int = 24) -> str:
    label = slug.replace("-", " ").strip()
    label = label[:1].upper() + label[1:]
    return (label[: limit - 1].rstrip() + "…") if len(label) > limit else label


def _flow_snippet_lines(snippet: str, max_lines: int = 5, limit: int = 44) -> List[str]:
    """Up to `max_lines` non-blank code lines (indentation preserved, each
    truncated to `limit`)."""
    out: List[str] = []
    for line in (snippet or "").splitlines():
        if not line.strip():
            continue
        text = line.rstrip()
        if len(text) > limit:
            text = text[: limit - 1].rstrip() + "…"
        out.append(text)
        if len(out) >= max_lines:
            break
    return out


def _wrap_two_lines(text: str, limit: int = 128) -> List[str]:
    if len(text) <= limit:
        return [text]
    words, line1, i = text.split(" "), "", 0
    while i < len(words) and len(line1) + len(words[i]) + 1 <= limit:
        line1 = (line1 + " " + words[i]).strip()
        i += 1
    line2 = " ".join(words[i:])
    if len(line2) > limit:
        line2 = line2[: limit - 1].rstrip() + "…"
    return [line1, line2]


def stepper_animation_svg(
    title: str, subtitle: str, filename: str, steps: List, interactive: bool = False, anchors=None
) -> str:
    """A 'Run an App'-style stepper animation. The editor on the left shows each
    step's full code while its node is highlighted on the right and prior nodes
    persist so the flow assembles. `steps` is a list of `(label, [code_line,...])`.

    `interactive=False` (default): loops forever via SMIL `<animate>` (works in an
    `<img>`). `interactive=True`: plays the steps once via an embedded `<script>`,
    freezes on the last, then advances one section per click — this requires the
    SVG to be embedded via `<object>` (so its script runs and clicks register).
    Returns "" for fewer than two steps."""
    steps = steps[:5]
    n = len(steps)
    if n < 2:
        return ""
    esc = lambda s: html.escape(str(s), quote=True)

    win = 0.90 / n
    intro = 0.06
    nx0, nx1, gap = 600, 1224, 36
    node_w = (nx1 - nx0 - gap * (n - 1)) / n
    node_h, node_y = 86, 250
    cy = node_y + node_h / 2.0
    code_y0, code_lh = 248, 26
    sub_lines = _wrap_two_lines(subtitle)

    # Looping path uses SMIL <animate> on opacity (runs in <img>; CSS animation
    # does not). Interactive path is script-driven and embedded via <object>.
    def anim(values, times):
        ts = ";".join(f"{max(0.0, min(1.0, t)):.3f}" for t in times)
        return (
            f'<animate attributeName="opacity" dur="12s" repeatCount="indefinite" '
            f'calcMode="linear" values="{values}" keyTimes="{ts}"/>'
        )

    # opacity + (SMIL animate | id) for a group, depending on mode.
    def group_open(kind, idx, smil_values, smil_times):
        if interactive:
            init = "1" if idx == 0 else "0"
            return [f'<g id="{kind}{idx}" class="fanim" opacity="{init}">']
        return ['<g opacity="0">', anim(smil_values, smil_times)]

    css = (
        '.fsans{font-family:"Avenir Next","Segoe UI",Arial,sans-serif;}'
        '.fmono{font-family:"SFMono-Regular","Menlo","Consolas",monospace;}'
        '.fttl{fill:#010F0E;font-size:27px;font-weight:700;}'
        '.fsub{fill:#35536F;font-size:15px;font-weight:500;}'
        '.fcode{fill:#E2ECF6;font-size:14px;font-weight:500;}'
        '.fln{fill:#586677;font-size:13px;font-weight:500;}'
        '.fcap{fill:#7FD0A6;font-size:14px;font-weight:700;letter-spacing:0.04em;}'
        '.fdim{fill:#7D8590;font-size:14px;font-weight:500;}'
        '.ftag{fill:#F8FBFF;font-weight:650;}'
        '.fnl{fill:#0B2E1B;font-weight:650;}'
        '.fcounter{fill:#9AA7B4;font-size:13px;font-weight:600;}'
        '.fhint{fill:#2A9C4F;font-size:15px;font-weight:700;}'
        '.fanim{transition:opacity 0.55s ease;}'
    )

    p: List[str] = []
    p.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1320 520" role="img" aria-labelledby="ftl ftd">')
    p.append(f'<title id="ftl">{esc(title)}</title>')
    p.append(f'<desc id="ftd">{esc(subtitle)}</desc>')
    p.append('<defs>')
    p.append('<linearGradient id="fbg" x1="80" y1="40" x2="1240" y2="500" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#F3F8F6"/></linearGradient>')
    p.append('<linearGradient id="fgreen" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2A9C4F"/><stop offset="1" stop-color="#C0E8DB"/></linearGradient>')
    p.append('<filter id="fsh" x="-10%" y="-18%" width="120%" height="142%" filterUnits="objectBoundingBox"><feDropShadow dx="0" dy="14" stdDeviation="16" flood-color="#010F0E" flood-opacity="0.12"/></filter>')
    p.append('<marker id="farr" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto" markerUnits="strokeWidth"><path d="M0 0L7 3.5L0 7Z" fill="#2A9C4F"/></marker>')
    p.append('<style>' + css + '</style>')
    p.append('</defs>')

    p.append('<rect x="16" y="16" width="1288" height="488" rx="30" fill="url(#fbg)" stroke="#D8E7F1"/>')
    p.append('<circle cx="1188" cy="82" r="58" fill="#C0E8DB" opacity="0.5"/>')
    p.append('<circle cx="112" cy="432" r="50" fill="#BFE0CF" opacity="0.45"/>')
    p.append(f'<text x="54" y="68" class="fsans fttl">{esc(title)}</text>')
    for k, sl in enumerate(sub_lines):
        p.append(f'<text x="54" y="{96 + k * 20}" class="fsans fsub">{esc(sl)}</text>')

    # editor: one code group per step
    p.append('<g filter="url(#fsh)">')
    p.append('<rect x="54" y="132" width="470" height="316" rx="26" fill="#1F232A"/>')
    p.append('<rect x="54" y="132" width="470" height="46" rx="26" fill="#181C22"/>')
    p.append('<circle cx="88" cy="155" r="6" fill="#FF6B6B"/><circle cx="110" cy="155" r="6" fill="#F4C542"/><circle cx="132" cy="155" r="6" fill="#2A9C4F"/>')
    p.append(f'<text x="166" y="160" class="fmono fdim">{esc(filename)}</text>')
    for i, (label, lines) in enumerate(steps):
        si = intro + i * win
        ei = 0.985 if i == n - 1 else si + win
        p.extend(group_open("code", i, "0;0;1;1;0;0", [0, si, si + 0.015, ei - 0.02, ei, 1]))
        p.append(f'<text x="76" y="212" class="fsans fcap">STEP {i + 1} · {esc(label)}</text>')
        for j, cl in enumerate(lines):
            by = code_y0 + j * code_lh
            p.append(f'<text x="76" y="{by}" class="fmono fln">{j + 1}</text>')
            p.append(f'<text x="104" y="{by}" class="fmono fcode" xml:space="preserve">{esc(cl)}</text>')
        p.append('</g>')
    p.append('</g>')

    # right: panel + nodes (assemble & persist), active node ringed
    p.append('<g filter="url(#fsh)">')
    if interactive:
        p.append('<g id="shell" opacity="1">')
    else:
        p.append('<g opacity="0">')
        p.append(anim("0;0;1;1;0", [0, 0.04, 0.06, 0.98, 1]))
    p.append('<rect x="560" y="132" width="704" height="316" rx="30" fill="#FFFFFF" stroke="#9CD5B2" stroke-width="2"/>')
    p.append('<rect x="592" y="158" width="160" height="38" rx="19" fill="url(#fgreen)"/><text x="672" y="183" class="fsans ftag" font-size="14" text-anchor="middle">Walkthrough</text>')
    p.append('</g>')
    fs = 15 if node_w >= 150 else (13 if node_w >= 120 else 11.5)
    for i, (label, _lines) in enumerate(steps):
        si = intro + i * win
        ei = 0.985 if i == n - 1 else si + win
        x = nx0 + i * (node_w + gap)
        cxn = x + node_w / 2.0
        last = i == n - 1
        fill = "#F8F6FF" if last else "#F3FBF7"
        stroke = "#B8B0DC" if last else "#9CD5B2"
        p.extend(group_open("node", i, "0;0;1;1;0", [0, si, si + 0.02, 0.98, 1]))
        if i > 0:
            p.append(
                f'<line x1="{x - gap + 4:.0f}" y1="{cy:.0f}" x2="{x - 6:.0f}" y2="{cy:.0f}" '
                'stroke="#2A9C4F" stroke-width="3" stroke-linecap="round" marker-end="url(#farr)"/>'
            )
        p.append(f'<rect x="{x:.0f}" y="{node_y}" width="{node_w:.0f}" height="{node_h}" rx="18" fill="{fill}" stroke="{stroke}"/>')
        p.append(f'<text x="{cxn:.0f}" y="{node_y + 22}" class="fsans fdim" font-size="12" text-anchor="middle">Step {i + 1}</text>')
        words = label.split(" ")
        if len(words) > 1 and node_w < 150:
            mid = (len(words) + 1) // 2
            l1, l2 = " ".join(words[:mid]), " ".join(words[mid:])
            p.append(
                f'<text x="{cxn:.0f}" y="{cy + 6:.0f}" class="fsans fnl" font-size="{fs}" text-anchor="middle">'
                f'<tspan x="{cxn:.0f}" dy="0">{esc(l1)}</tspan><tspan x="{cxn:.0f}" dy="{fs + 2:.0f}">{esc(l2)}</tspan></text>'
            )
        else:
            p.append(f'<text x="{cxn:.0f}" y="{cy + fs / 3.0:.0f}" class="fsans fnl" font-size="{fs}" text-anchor="middle">{esc(label)}</text>')
        p.append('</g>')
        p.extend(group_open("ring", i, "0;0;1;1;0;0", [0, si + 0.005, si + 0.025, ei - 0.02, ei, 1]))
        p.append(
            f'<rect x="{x - 4:.0f}" y="{node_y - 4}" width="{node_w + 8:.0f}" '
            f'height="{node_h + 8}" rx="22" fill="none" stroke="#2A9C4F" stroke-width="3"/>'
        )
        p.append('</g>')
    p.append('</g>')

    if interactive:
        # Popup revealed after the play-through, prompting the reader to click a step.
        p.append('<g id="popup" class="fanim" opacity="0">')
        p.append('<rect x="690" y="356" width="444" height="46" rx="23" fill="#0B2E1B" opacity="0.93"/>')
        p.append(
            '<text x="912" y="384" text-anchor="middle" class="fsans" fill="#EAF7EF" '
            'font-size="16" font-weight="700">Click a step to jump to that section ↓</text>'
        )
        p.append('</g>')
        p.append(
            _STEPPER_SCRIPT.replace("__N__", str(n)).replace("__ANCHORS__", json.dumps(anchors or []))
        )

    p.append('</svg>')
    return "\n".join(p)


# Drives the interactive stepper: play each step once (assembling the flow),
# then reveal the popup and make each step node clickable — a click scrolls the
# parent tutorial page to that step's `#step-<name>` section. Only runs when the
# SVG is embedded via <object> (an <img>-embedded SVG never executes scripts).
_STEPPER_SCRIPT = """<script type="text/ecmascript"><![CDATA[
(function(){
  var n=__N__, i=0, t=null, anchors=__ANCHORS__;
  function el(id){return document.getElementById(id);}
  function paint(a){
    for(var k=0;k<n;k++){
      var c=el("code"+k); if(c)c.style.opacity=(k===a?"1":"0");
      var r=el("ring"+k); if(r)r.style.opacity=(k===a?"1":"0");
      var d=el("node"+k); if(d)d.style.opacity=(k<=a?"1":"0");
    }
  }
  function finish(){
    for(var k=0;k<n;k++){ var d=el("node"+k); if(d)d.style.opacity="1"; var r=el("ring"+k); if(r)r.style.opacity="0"; }
    var pop=el("popup"); if(pop)pop.style.opacity="1";
  }
  function tick(){ paint(i); if(i>=n-1){ finish(); return; } i++; t=setTimeout(tick,2600); }
  function go(a){
    var name=anchors[a]; if(!name)return;
    try{
      var pdoc=window.parent.document, tgt=pdoc.getElementById("step-"+name);
      if(tgt){ tgt.scrollIntoView({behavior:"smooth",block:"start"}); if(window.parent.history&&window.parent.history.replaceState){window.parent.history.replaceState(null,"","#step-"+name);} return; }
    }catch(e){}
    try{ window.parent.location.hash="step-"+name; }catch(e2){ try{ window.top.location.hash="step-"+name; }catch(e3){} }
  }
  for(var k=0;k<n;k++){ (function(idx){ var d=el("node"+idx); if(d&&d.addEventListener){ d.style.cursor="pointer"; d.addEventListener("click",function(){go(idx);}); } })(k); }
  paint(0);
  t=setTimeout(tick,650);
})();
]]></script>"""


def flow_animation_svg(module: TutorialModule) -> str:
    """Per-tutorial stepper animation built from its Walkthrough steps."""
    steps = [s for s in module.walkthrough_steps if s.name and (s.py_snippet or s.cpp_snippet)]
    if len(steps) < 2:
        return ""
    chosen = steps[:5]
    data = [
        (_flow_node_label(s.name), _flow_snippet_lines(s.py_snippet or s.cpp_snippet))
        for s in chosen
    ]
    anchors = [s.name for s in chosen]
    title = re.sub(r"\s+", " ", module.display_title).strip()
    subtitle = _first_sentence(module.concept or module.walkthrough_lead)
    return stepper_animation_svg(
        title, subtitle, pathlib.Path(module.py_rel).name, data, interactive=True, anchors=anchors
    )


def generate_flow_animations(out_dir: pathlib.Path, modules: List[TutorialModule]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = set()
    for module in modules:
        svg = flow_animation_svg(module)
        if not svg:
            continue
        (out_dir / f"{module.folder}.svg").write_text(svg, encoding="utf-8")
        expected.add(f"{module.folder}.svg")
    for stale in out_dir.glob("*.svg"):
        if stale.name not in expected:
            stale.unlink(missing_ok=True)


def generate_landing_animation(out_dir: pathlib.Path) -> None:
    """Looping animated hero for the tutorials index (/tutorials): the three-stage
    journey the chapters take you through. Written outside the per-tutorial
    `flow/` dir so that directory's stale-file cleanup doesn't sweep it."""
    steps = [
        ("Run a model", ["model = pyneat.Model('model.tar.gz')", "sample = model.run([image])"]),
        ("Build a pipeline", ["graph = pyneat.Graph()", "graph.add(model)", "run = graph.build([frame])"]),
        ("Tune & deploy", ["opt.queue_depth = 8", "run = graph.build([f], Async, opt)", "metrics = run.stats()"]),
    ]
    svg = stepper_animation_svg(
        "Tutorials",
        "Guided C++ and Python tutorials, from your first model to a production pipeline.",
        "your_app.py",
        steps,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "landing.svg").write_text(svg, encoding="utf-8")


def generate_minimal_example_animation(repo_root: pathlib.Path) -> None:
    """Stepper animation for the hand-authored 'Minimal' getting-started page
    (write → run → confirm). Written as a relative docs asset so the Markdown
    image reference stays baseUrl-safe, like the Run an App animation."""
    steps = [
        (
            "Write the script",
            [
                "from pyneat import DeviceType",
                "def main():",
                '    print("Hello from sima-neat")',
                '    print("DeviceType.CPU =", DeviceType.CPU)',
            ],
        ),
        ("Run on the DevKit", ["source ~/pyneat/bin/activate", "python3 hello_neat.py"]),
        ("Runtime responds", ["Hello from sima-neat", "DeviceType.CPU = DeviceType.CPU"]),
    ]
    svg = stepper_animation_svg(
        "Minimal: confirm your install",
        "After installation, confirm your Neat setup is wired correctly.",
        "hello_neat.py",
        steps,
    )
    out = repo_root / "docs" / "images" / "minimal-example-flow.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")


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
    in_practice_section = _extract_section(text, "In Practice")

    meta = _parse_metadata_table(meta_section)
    difficulty = meta.get("difficulty", "Intermediate")
    estimated = meta.get("estimated read time", "10-15 minutes")
    labels_raw = meta.get("labels", "")
    labels = [x.strip() for x in labels_raw.split(",") if x.strip()]

    process_steps = _parse_numbered(process_section)

    cpp_rel = cpp.resolve().relative_to(repo_root.resolve()).as_posix()
    py_rel = py.resolve().relative_to(repo_root.resolve()).as_posix()

    walkthrough_lead, walkthrough_steps = _build_walk_steps(
        text,
        cpp.read_text(encoding="utf-8"),
        py.read_text(encoding="utf-8"),
        name,
    )

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
        in_practice=in_practice_section,
        cpp_rel=cpp_rel,
        py_rel=py_rel,
        walkthrough_lead=walkthrough_lead,
        walkthrough_steps=walkthrough_steps,
    )


def _full_code_codetabs(module: TutorialModule) -> List[str]:
    """A `<CodeTabs>` block with the complete C++ and Python files (CORE LOGIC
    highlight preserved). Used by the legacy `## Code` section and by the
    walkthrough's collapsed "Full source" block."""
    cpp_src = pathlib.Path(module.cpp_rel).read_text(encoding="utf-8").rstrip()
    py_src = pathlib.Path(module.py_rel).read_text(encoding="utf-8").rstrip()
    cpp_code, cpp_hl = _render_code_with_core_logic(cpp_src)
    py_code, py_hl = _render_code_with_core_logic(py_src)
    return [
        "<CodeTabs>",
        '<CodeTab label="C++" lang="cpp">',
        "",
        _code_fence("cpp", module.cpp_rel, cpp_hl),
        cpp_code,
        "```",
        "",
        "</CodeTab>",
        '<CodeTab label="Python" lang="python">',
        "",
        _code_fence("python", module.py_rel, py_hl),
        py_code,
        "```",
        "",
        "</CodeTab>",
        "</CodeTabs>",
    ]


def _step_codetabs(module: TutorialModule, step: WalkStep) -> List[str]:
    """A short, un-highlighted `<CodeTabs>` block for one walkthrough step, with
    only the language snippets that exist. Highlighting is reserved for the
    collapsed full-source listing; a per-step snippet is already the excerpt."""
    out: List[str] = ["<CodeTabs>"]
    if step.cpp_snippet:
        out.extend(
            [
                '<CodeTab label="C++" lang="cpp">',
                "",
                _code_fence("cpp", module.cpp_rel, ""),
                step.cpp_snippet,
                "```",
                "",
                "</CodeTab>",
            ]
        )
    if step.py_snippet:
        out.extend(
            [
                '<CodeTab label="Python" lang="python">',
                "",
                _code_fence("python", module.py_rel, ""),
                step.py_snippet,
                "```",
                "",
                "</CodeTab>",
            ]
        )
    out.append("</CodeTabs>")
    return out


def _step_language_prose(step: WalkStep) -> List[str]:
    """Language-specific prose wrapped in `<LanguageContent>` so it toggles in
    lockstep with the code language (same `neat-docs-language` preference the
    `<CodeTabs>` use). Empty when the step has no per-language prose."""
    out: List[str] = []
    for lang_attr, prose in (("cpp", step.cpp_prose), ("py", step.py_prose)):
        if prose:
            out.extend(
                ["", f'<LanguageContent lang="{lang_attr}">', "", prose, "", "</LanguageContent>"]
            )
    return out


_RUN_PY_LABEL = "**Python:**"
_RUN_CPP_LABELS = ("**C++ (prebuilt):**", "**C++ (build from source):**")


def _strip_blank_edges(block: List[str]) -> List[str]:
    while block and not block[0].strip():
        block = block[1:]
    while block and not block[-1].strip():
        block = block[:-1]
    return block


def _wrap_run_languages(run_text: str) -> str:
    """Separate the Python and C++ command blocks in a `## Run` section behind
    the language toggle: the `**Python:**` block is wrapped in
    `<LanguageContent lang="py">` and the two `**C++ …**` blocks in
    `<LanguageContent lang="cpp">`, so a reader sees only their language's
    commands. Shared intro/expected-output prose stays outside the wrappers.
    Returns the text unchanged if it doesn't match the expected shape."""
    lines = run_text.split("\n")

    def label_index(label: str) -> int:
        for i, line in enumerate(lines):
            if line.strip() == label:
                return i
        return -1

    py_i = label_index(_RUN_PY_LABEL)
    cpp1_i = label_index(_RUN_CPP_LABELS[0])
    cpp2_i = label_index(_RUN_CPP_LABELS[1])
    if not (0 <= py_i < cpp1_i < cpp2_i):
        return run_text  # unexpected shape — pass through untouched

    # The last C++ block ends at the closing ``` of its single fenced example.
    j = cpp2_i + 1
    while j < len(lines) and not lines[j].lstrip().startswith("```"):
        j += 1
    j += 1  # past the opening fence
    while j < len(lines) and not lines[j].lstrip().startswith("```"):
        j += 1
    if j >= len(lines):
        return run_text  # no closing fence found — don't risk a bad split
    cpp_end = j  # index of the closing ``` line

    intro = _strip_blank_edges(lines[:py_i])
    py_block = _strip_blank_edges(lines[py_i:cpp1_i])
    cpp_block = _strip_blank_edges(lines[cpp1_i : cpp_end + 1])
    trailing = _strip_blank_edges(lines[cpp_end + 1 :])

    out: List[str] = list(intro)
    out += ["", '<LanguageContent lang="py">', "", *py_block, "", "</LanguageContent>"]
    out += ["", '<LanguageContent lang="cpp">', "", *cpp_block, "", "</LanguageContent>"]
    if trailing:
        out += ["", *trailing]
    return "\n".join(out)


def render_legacy_body(module: TutorialModule) -> List[str]:
    """The original page body: Concept, Learning Process, Run, In Practice, and
    one full-file Code listing. Used for tutorials without a Walkthrough."""
    lines: List[str] = [
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

    # Pass the README's `## Run` section through verbatim so the README is the
    # single source of truth for run instructions — no hardcoded env-var blocks.
    if module.run_section.strip():
        lines.extend(["", "## Run", "", _wrap_run_languages(module.run_section)])
    # Pass the README's optional `## In Practice` section through verbatim so
    # operational guidance (folded from the former How-To guides) is taught in
    # context. Markdown tables pass through like the `## Run` block above.
    if module.in_practice.strip():
        lines.extend(["", "## In Practice", "", module.in_practice])
    lines.extend(["", "## Code", ""])
    lines.extend(_full_code_codetabs(module))
    return lines


def render_walkthrough_body(module: TutorialModule) -> List[str]:
    """The narrative page body: an up-front lead, then a sequence of short
    explained code steps, then Run, optional In Practice, and a collapsed
    full-source block."""
    lines: List[str] = []
    if module.walkthrough_lead:
        lines.extend(["", module.walkthrough_lead])
    lines.append("")
    lines.append("## Walkthrough")
    for step in module.walkthrough_steps:
        anchor = f" {{#step-{step.name}}}" if step.name else ""
        lines.extend(["", f"### {step.heading}{anchor}", "", step.shared_prose])
        lines.extend(_step_language_prose(step))
        if step.cpp_snippet or step.py_snippet:
            lines.append("")
            lines.extend(_step_codetabs(module, step))

    # End the narrative with running it and observing the expected output. The
    # README's `## Run` section ends with the expected stdout.
    if module.run_section.strip():
        lines.extend(["", "## Run", "", _wrap_run_languages(module.run_section)])
    if module.in_practice.strip():
        lines.extend(["", "## In Practice", "", module.in_practice])

    # Keep the complete programs available without letting them dominate the
    # page — collapsed by default so the per-step snippets stay the focus.
    lines.extend(
        [
            "",
            "## Full source",
            "",
            "<details>",
            "<summary>Show the complete C++ and Python programs</summary>",
            "",
        ]
    )
    lines.extend(_full_code_codetabs(module))
    lines.extend(["", "</details>"])
    return lines


def render_tutorial_doc(module: TutorialModule, sidebar_position: int, repo_ref: str) -> str:
    repo_link_prefix = f"{REPO_LINK_BASE}/{repo_ref}"
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
    ]

    # Animated overview directly under the title. Embedded via <object> (not
    # <img>) so the SVG's script runs — it plays once, then steps on click.
    # The <img> fallback covers no-JS / object-load failures.
    if flow_animation_svg(module):
        alt = html.escape(f"{module.display_title} — animated walkthrough overview", quote=True)
        url = module.flow_image_url
        lines.append(
            f'<object type="image/svg+xml" data="{url}" class="tutorial-flow" aria-label="{alt}">'
            f'<img src="{url}" alt="{alt}" loading="lazy" /></object>'
        )
        lines.append("")

    lines.extend(
        [
            "| Field | Value |",
            "| --- | --- |",
            f"| Difficulty | {module.difficulty} |",
            f"| Estimated Read Time | {module.estimated_read_time} |",
            f"| Labels | {label_text} |",
            "",
        ]
    )

    if module.has_walkthrough:
        lines.extend(render_walkthrough_body(module))
    else:
        lines.extend(render_legacy_body(module))

    lines.extend(
        [
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


def _group_tutorials(modules: List[TutorialModule]) -> Dict[str, List[TutorialModule]]:
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
    return groups


def _render_tutorial_card_grid(modules: List[TutorialModule]) -> List[str]:
    lines: List[str] = ['<div class="tutorial-grid">']
    for module in modules:
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
    return lines


def _clean_heading_body(heading_body: str) -> str:
    return heading_body.replace(
        '<p class="tutorial-grid-intro">Use these tutorials in order. Each card links to a chapter with concept-first guidance and matching C++ and Python implementation.</p>',
        "",
    ).strip()


def _render_tutorial_path_block(groups: Dict[str, List[TutorialModule]]) -> List[str]:
    lines: List[str] = [
        '<div class="overview-link-columns">',
        '  <section class="overview-link-panel overview-link-panel-start">',
        "    <h2>Choose a Tutorial Path</h2>",
        "    <p>Use the cards in each section in order. Each tutorial includes concept-first guidance with matching C++ and Python implementation.</p>",
        '    <ul class="overview-link-list">',
    ]
    difficulty_copy = {
        "Beginner": "First model run, async inference, model benchmarking, basic graphs, and model options.",
        "Intermediate": "Data exchange, preprocessing, outputs, streaming, diagnostics, and graph composition.",
        "Advanced": "Multi-stream graphs, throughput tuning, and production-style pipeline structure.",
    }
    for difficulty, slug, _ in DIFFICULTY_SUBDIRS:
        count = len(groups[difficulty])
        lines.append(
            f'      <li><a class="overview-link-card" href="/tutorials/{slug}/"><strong>{difficulty}</strong><span>{difficulty_copy[difficulty]} {count} guided chapters.</span></a></li>'
        )
    lines.extend(["    </ul>", "  </section>", "</div>", ""])
    return lines


def render_index(modules: List[TutorialModule], heading_body: str) -> str:
    groups = _group_tutorials(modules)
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
        "",
        f'<img src="{docs_static_url("img/tutorials/landing.svg")}" alt="Neat tutorials — from your first model to a production pipeline" class="tutorial-flow" loading="lazy" />',
    ]

    cleaned_heading_body = _clean_heading_body(heading_body)
    if cleaned_heading_body:
        lines.extend(["", cleaned_heading_body, ""])
    lines.extend(_render_tutorial_path_block(groups))
    return "\n".join(lines)


def render_difficulty_index(difficulty: str, slug: str, modules: List[TutorialModule]) -> str:
    lines: List[str] = [
        "---",
        f"title: {difficulty}",
        f"description: {difficulty} Neat tutorials",
        "sidebar_position: 1",
        f"slug: /tutorials/{slug}",
        "---",
        "",
        "<!-- AUTO-GENERATED by tools/generate_tutorial_docs.py. -->",
        "",
        f"# {difficulty} Tutorials",
        "",
        '<p class="tutorial-grid-intro">Use these tutorials in order. Each card links to a chapter with concept-first guidance and matching C++ and Python implementation.</p>',
        "",
    ]
    lines.extend(_render_tutorial_card_grid(modules))
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
    docs_tutorials_dir = root / "docs" / "develop-apps" / "tutorials"

    module_dirs = discover_modules(tutorials_dir)
    modules = [parse_module(d, root) for d in module_dirs]
    modules.sort(key=lambda m: _flow_key(m.number))
    repo_ref = detect_repo_ref()

    docs_tutorials_dir.mkdir(parents=True, exist_ok=True)
    generate_card_images(root / "website" / "static" / "img" / "tutorials" / "cards")
    generate_flow_animations(root / "website" / "static" / "img" / "tutorials" / "flow", modules)
    generate_landing_animation(root / "website" / "static" / "img" / "tutorials")
    generate_minimal_example_animation(root)
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

    grouped_modules = _group_tutorials(modules)

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

    for label, slug, _ in DIFFICULTY_SUBDIRS:
        out_path = docs_tutorials_dir / slug / "index.md"
        out_path.write_text(
            render_difficulty_index(label, slug, grouped_modules[label]),
            encoding="utf-8",
        )

    index_path = docs_tutorials_dir / "index.md"
    index_path.write_text(render_index(modules, heading_body), encoding="utf-8")

    print(f"Generated {len(modules)} tutorial docs + index (source ref: {repo_ref}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    repo_link_prefix = f"{REPO_LINK_BASE}/{repo_ref}"
