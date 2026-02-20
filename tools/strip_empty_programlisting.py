#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XML_DIR = ROOT / "docs" / "doxygen" / "out" / "xml"

EMPTY_LISTING_RE = re.compile(r"<programlisting\b[^>]*>\s*</programlisting>", re.MULTILINE)


def main() -> None:
    if not XML_DIR.exists():
        return
    for xml in XML_DIR.rglob("*.xml"):
        text = xml.read_text()
        new_text = EMPTY_LISTING_RE.sub("", text)
        if new_text != text:
            xml.write_text(new_text)


if __name__ == "__main__":
    main()
