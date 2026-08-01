import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "autodoc.py"
SPEC = importlib.util.spec_from_file_location("autodoc", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PromoteIndexFileTests(unittest.TestCase):
    def test_rewrites_inbound_markdown_and_html_links(self):
        with tempfile.TemporaryDirectory() as directory:
            section = Path(directory)
            (section / "README.md").write_text("# Landing\n", encoding="utf-8")
            nested = section / "panels" / "power.md"
            nested.parent.mkdir()
            nested.write_text(
                "[Overview](../README.md#overview)\n"
                '<a href="../README.md?view=full">HTML overview</a>\n'
                "[Unrelated](../reports.md)\n",
                encoding="utf-8",
            )

            MODULE.promote_index_file({"index_file": "README.md"}, section)

            self.assertFalse((section / "README.md").exists())
            self.assertTrue((section / "index.md").exists())
            rewritten = nested.read_text(encoding="utf-8")
            self.assertIn("[Overview](../index.md#overview)", rewritten)
            self.assertIn('href="../index.md?view=full"', rewritten)
            self.assertIn("[Unrelated](../reports.md)", rewritten)


if __name__ == "__main__":
    unittest.main()
