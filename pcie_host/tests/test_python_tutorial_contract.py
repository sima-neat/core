#!/usr/bin/env python3
"""Keep shipped Python tutorials on the public PyNeatPCIe surface."""

import ast
import unittest
from pathlib import Path


TUTORIALS = Path(__file__).resolve().parents[1] / "tutorials"


def attribute_root(node):
  while isinstance(node, ast.Attribute):
    node = node.value
  return node.id if isinstance(node, ast.Name) else None


class PythonTutorialContract(unittest.TestCase):
  def test_tutorials_compile_and_use_only_public_pyneatpcie_names(self):
    scripts = sorted(TUTORIALS.glob("[0-9][0-9][0-9]_*/*.py"))
    self.assertGreaterEqual(len(scripts), 9)

    for script in scripts:
      with self.subTest(script=script.relative_to(TUTORIALS)):
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(script))
        compile(tree, str(script), "exec")

        for node in ast.walk(tree):
          if isinstance(node, ast.ImportFrom) and node.module:
            self.assertFalse(
                node.module.startswith("pyneatpcie."),
                f"{script.name} imports a PyNeatPCIe implementation module",
            )
          if isinstance(node, ast.Attribute) and attribute_root(node) == "pcie":
            self.assertFalse(
                node.attr.startswith("_"),
                f"{script.name} uses private PyNeatPCIe name {node.attr}",
            )


if __name__ == "__main__":
  unittest.main()
