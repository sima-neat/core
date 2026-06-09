#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XML_DIR = ROOT / "docs" / "doxygen" / "out" / "xml"

EMPTY_LISTING_RE = re.compile(r"<programlisting\b[^>]*>\s*</programlisting>", re.MULTILINE)
HEADER_LISTING_RE = re.compile(r"\s*<programlisting\b[^>]*>.*?</programlisting>", re.DOTALL)


def main() -> None:
    if not XML_DIR.exists():
        return
    for xml in XML_DIR.rglob("*.xml"):
        text = xml.read_text()
        if xml.name.endswith(("_8h.xml", "_8hpp.xml", "_8hh.xml", "_8hxx.xml")):
            # Header file pages are public reference docs, not source browsers.  Removing
            # program listings avoids surfacing internal-only #ifdef blocks and links to
            # headers intentionally kept out of the installed/public API.
            new_text = HEADER_LISTING_RE.sub("", text)
        else:
            new_text = EMPTY_LISTING_RE.sub("", text)
        if new_text != text:
            xml.write_text(new_text)


if __name__ == "__main__":
    main()
