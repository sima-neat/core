#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "docs" / "reference" / "cppapi"

SECTIONS = ["files", "structs", "classes", "namespaces", "groups"]

link_re = re.compile(
    r"(/PipelineSession//reference/api/|/PipelineSession/reference/api/|/PipelineSession/docs/reference/api/|/docs/reference/api/|/reference/api/|"
    r"/PipelineSession//reference/cppapi/|/PipelineSession/reference/cppapi/|/PipelineSession/docs/reference/cppapi/|/docs/reference/cppapi/|/reference/cppapi/)"
    r"(?P<section>files|structs|classes|namespaces|groups)"
    r"/(?P<path>[A-Za-z0-9_./-]+)"
)

dash_anchor_re = re.compile(r"(/PipelineSession/(docs/)?reference/(api|cppapi)/[A-Za-z0-9_/.-]+|/reference/(api|cppapi)/[A-Za-z0-9_/.-]+)-#")
docs_prefix_re = re.compile(r"/PipelineSession/docs/reference/(api|cppapi)/")
docs_root_prefix_re = re.compile(r"/docs/reference/(api|cppapi)/")
pipeline_prefix_re = re.compile(r"/PipelineSession/reference/(api|cppapi)/")
pipeline_root_re = re.compile(r"/PipelineSession//reference/(api|cppapi)/")

slug_re = re.compile(r"^slug:\s*/reference/(api|cppapi)/(?P<section>files|structs|classes|namespaces)/(?P<path>[A-Za-z0-9_./-]+)\s*$", re.MULTILINE)


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
        if new_text != text:
            md.write_text(new_text)


if __name__ == "__main__":
    main()
