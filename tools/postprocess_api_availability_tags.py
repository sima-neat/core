#!/usr/bin/env python3
"""Render API availability tags in generated Markdown docs.

Supported tags in source comments/docstrings:
  <since>2.1.0</since>
  <compatible-with>>=2.1.0</compatible-with>

The script rewrites tag occurrences to styled HTML badges so docs show:
  Since 2.1.0
  Compatible With >=2.1.0
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


SINCE_RE = re.compile(r"<since>\s*([^<]+?)\s*</since>", re.IGNORECASE)
COMPAT_RE = re.compile(
    r"<compatible-with>\s*([^<]+?)\s*</compatible-with>", re.IGNORECASE
)
SINCE_ESC_RE = re.compile(
    r"&lt;since&gt;\s*([^<]+?)\s*&lt;/since&gt;", re.IGNORECASE
)
COMPAT_ESC_RE = re.compile(
    r"&lt;compatible-with&gt;\s*([^<]+?)\s*&lt;/compatible-with&gt;",
    re.IGNORECASE,
)


def _safe_badge_text(text: str) -> str:
    # Keep '>' readable (e.g., ">=2.1.0"), escape only unsafe chars.
    text = html.unescape(text.strip())
    return text.replace("&", "&amp;").replace("<", "&lt;")


def _since_badge(version: str) -> str:
    safe = _safe_badge_text(version)
    return (
        f'<span class="api-availability-badge api-since">Since {safe}</span>'
    )


def _compat_badge(spec: str) -> str:
    safe = _safe_badge_text(spec)
    return (
        f'<span class="api-availability-badge api-compatible">Compatible With {safe}</span>'
    )


def transform_text(text: str) -> str:
    text = SINCE_RE.sub(lambda m: _since_badge(m.group(1)), text)
    text = COMPAT_RE.sub(lambda m: _compat_badge(m.group(1)), text)
    text = SINCE_ESC_RE.sub(lambda m: _since_badge(m.group(1)), text)
    text = COMPAT_ESC_RE.sub(lambda m: _compat_badge(m.group(1)), text)

    # When a normal description paragraph is immediately followed by an
    # availability badge paragraph, render badges first for quicker scanning.
    text = re.sub(
        r'(<p>(?!<span class="api-availability-badge")(?!Definition at line)'
        r'(?:(?!</p>).)*</p>)'
        r'(\s*<p><span class="api-availability-badge(?:(?!</p>).)*</p>)',
        lambda m: f"{m.group(2)}\n\n{m.group(1)}",
        text,
        flags=re.DOTALL,
    )

    # Trim awkward spacing around injected badges.
    text = re.sub(r"[ \t]+(<span class=\"api-availability-badge)", r" \1", text)
    text = re.sub(r"(</span>)[ \t]+", r"\1 ", text)
    return text


def process_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = transform_text(original)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs-dir",
        required=True,
        help="Directory containing generated markdown docs to rewrite",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        raise SystemExit(f"docs directory not found: {docs_dir}")

    changed = 0
    for md in sorted(docs_dir.rglob("*.md")):
        if process_file(md):
            changed += 1

    print(f"Processed availability tags in {changed} markdown file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
