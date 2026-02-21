#!/usr/bin/env python3
"""Copy docs to an expanded location without rewriting MDX CodeTabs.

Historically this script rewrote <CodeTabs>/<CodeTab> into HTML markers used by
legacy DOM-mutation JavaScript. We now use React MDX components for tabs, so
the script intentionally performs a plain directory copy.
"""

from __future__ import annotations

import argparse
import os
import shutil


def copy_tree(src: str, dst: str) -> None:
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy docs tree to expanded path.")
    parser.add_argument("--src", required=True, help="Source docs directory.")
    parser.add_argument("--dst", required=True, help="Destination docs directory.")
    args = parser.parse_args()

    copy_tree(args.src, args.dst)


if __name__ == "__main__":
    main()
