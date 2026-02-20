#!/usr/bin/env python3
import argparse
import os
import re
import shutil
from pathlib import Path


TAB_BLOCK_RE = re.compile(r"<CodeTabs>(.*?)</CodeTabs>", re.DOTALL)
TAB_ITEM_RE = re.compile(
    r"<CodeTab\s+([^>]+)>(.*?)</CodeTab>",
    re.DOTALL,
)
ATTR_RE = re.compile(r'([a-zA-Z_-]+)\s*=\s*"(.*?)"')
CODE_FENCE_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+-]+)\n(?P<body>.*?)```",
    re.DOTALL,
)

LANG_LABELS = {
    "cpp": "C++",
    "c++": "C++",
    "cc": "C++",
    "cxx": "C++",
    "python": "Python",
    "py": "Python",
}

LANG_NORMALIZE = {
    "cpp": "cpp",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "python": "py",
    "py": "py",
}

LANG_CODE_CLASS = {
    "cpp": "cpp",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "python": "python",
    "py": "python",
}


def _escape_html(text):
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_tabs(code_blocks):
    if not code_blocks:
        return ""
    default_lang = LANG_NORMALIZE.get(code_blocks[0]["lang"], code_blocks[0]["lang"])
    tab_values = []
    fences = []
    for block in code_blocks:
        lang = block["lang"]
        tab_value = LANG_NORMALIZE.get(lang, lang)
        tab_values.append(tab_value)
        fences.append(f"```{lang}\n{block['body']}```")
    tabs_attr = ",".join(tab_values)
    return (
        f'<div class="code-tabs" data-default="{default_lang}" data-tabs="{tabs_attr}"></div>\n\n'
        + "\n\n".join(fences)
        + "\n\n<div class=\"code-tabs-end\"></div>\n\n"
    )


def _parse_attrs(attr_text):
    attrs = {}
    for match in ATTR_RE.finditer(attr_text):
        attrs[match.group(1)] = match.group(2)
    return attrs


def expand_content(content):
    def _replace(match):
        inner = match.group(1)
        blocks = []
        for tab_match in TAB_ITEM_RE.finditer(inner):
            attrs = _parse_attrs(tab_match.group(1))
            tab_body = tab_match.group(2)
            code_match = CODE_FENCE_RE.search(tab_body)
            if not code_match:
                continue
            lang = attrs.get("lang") or code_match.group("lang")
            lang = lang.strip().lower()
            body = code_match.group("body")
            body = body.lstrip("\n")
            if not body.endswith("\n"):
                body += "\n"
            blocks.append({"lang": lang, "body": body})
        return render_tabs(blocks) if blocks else match.group(0)

    return TAB_BLOCK_RE.sub(_replace, content)


def copy_tree(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def expand_in_dir(dst):
    for path in Path(dst).rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in (".md", ".mdx"):
            continue
        content = path.read_text(encoding="utf-8")
        if "<CodeTabs>" not in content:
            continue
        expanded = expand_content(content)
        if path.suffix == ".mdx":
            new_path = path.with_suffix(".md")
            new_path.write_text(expanded, encoding="utf-8")
            path.unlink()
        else:
            path.write_text(expanded, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Expand <CodeTabs> blocks.")
    parser.add_argument("--src", required=True, help="Source docs directory.")
    parser.add_argument("--dst", required=True, help="Destination docs directory.")
    args = parser.parse_args()

    copy_tree(args.src, args.dst)
    expand_in_dir(args.dst)


if __name__ == "__main__":
    main()
