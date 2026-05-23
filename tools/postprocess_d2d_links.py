#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "docs" / "reference" / "cppapi"

SECTIONS = ["files", "structs", "classes", "namespaces", "groups"]

link_re = re.compile(
    r"(//reference/api/|/reference/api/|/docs/reference/api/|/docs/reference/api/|/reference/api/|"
    r"//reference/cppapi/|/reference/cppapi/|/docs/reference/cppapi/|/docs/reference/cppapi/|/reference/cppapi/)"
    r"(?P<section>files|structs|classes|namespaces|groups)"
    r"/(?P<path>[A-Za-z0-9_./-]+)"
)

dash_anchor_re = re.compile(r"(/(docs/)?reference/(api|cppapi)/[A-Za-z0-9_/.-]+|/reference/(api|cppapi)/[A-Za-z0-9_/.-]+)-#")
docs_prefix_re = re.compile(r"/docs/reference/(api|cppapi)/")
docs_root_prefix_re = re.compile(r"/docs/reference/(api|cppapi)/")
pipeline_prefix_re = re.compile(r"/reference/(api|cppapi)/")
pipeline_root_re = re.compile(r"//reference/(api|cppapi)/")

slug_re = re.compile(r"^slug:\s*/reference/(api|cppapi)/(?P<section>files|structs|classes|namespaces)/(?P<path>[A-Za-z0-9_./-]+)\s*$", re.MULTILINE)

# doxygen2docusaurus emits anchors to /reference/cppapi/pages/deprecated/ and
# to groups.dox (a Doxygen-only source filename) that have no corresponding
# rendered page — we strip generate_api_docs.sh's "pages/" output and groups.dox
# is never published. Strip the dead hrefs so linkinator doesn't 404 on them,
# keeping the visible label text intact.
dead_anchor_re = re.compile(
    r'<a\s+href=(?:'
    r'"[^"]*(?:pages/deprecated/|groups\.dox)[^"]*"'
    r"|'[^']*(?:pages/deprecated/|groups\.dox)[^']*'"
    r'|[^\s>]*(?:pages/deprecated/|groups\.dox)[^\s>]*'
    r')\s*>(?P<label>.*?)</a>',
    re.DOTALL,
)


def normalize_link(match: re.Match) -> str:
    base = "/reference/cppapi/"
    section = match.group("section")
    path = match.group("path")
    # convert nested paths to flat slug with hyphens
    path = path.replace("/", "-")
    return f"{base}{section}/{path}"


def normalize_slug(text: str) -> str:
    def _repl(match: re.Match) -> str:
        section = match.group("section")
        path = match.group("path").replace("/", "-")
        return f"slug: /reference/cppapi/{section}/{path}"

    return slug_re.sub(_repl, text)


def main() -> None:
    if not API_DIR.exists():
        return
    for md in API_DIR.rglob("*.md"):
        text = md.read_text()
        # normalize double slashes and flatten paths
        new_text = link_re.sub(normalize_link, text)
        new_text = dash_anchor_re.sub(r"\1#", new_text)
        new_text = docs_prefix_re.sub("/reference/cppapi/", new_text)
        new_text = docs_root_prefix_re.sub("/reference/cppapi/", new_text)
        new_text = pipeline_prefix_re.sub("/reference/cppapi/", new_text)
        new_text = pipeline_root_re.sub("/reference/cppapi/", new_text)
        new_text = normalize_slug(new_text)
        new_text = dead_anchor_re.sub(lambda m: m.group("label"), new_text)
        if new_text != text:
            md.write_text(new_text)


if __name__ == "__main__":
    main()
