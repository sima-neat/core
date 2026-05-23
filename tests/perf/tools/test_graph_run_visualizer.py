#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from test_graph_run_schema import valid_payload


class GraphRunVisualizerTest(unittest.TestCase):
    def test_visualizer_writes_offline_html(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "graph_run.json"
            out = root / "graph_run.html"
            source.write_text(json.dumps(valid_payload()))
            subprocess.check_call(
                [sys.executable, str(ROOT / "tools" / "visualize_graph_run.py"), str(source), "-o", str(out)]
            )
            body = out.read_text()
            self.assertIn("<svg", body)
            self.assertIn("image", body)
            self.assertIn("classes", body)
            self.assertNotIn("unpkg.com", body)


if __name__ == "__main__":
    unittest.main()
