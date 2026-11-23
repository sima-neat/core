#!/usr/bin/env python3
"""Keep shipped PCIe tutorials on the public PyNeatPCIe surface and re-runnable."""

import ast
import unittest
from pathlib import Path


TUTORIALS = Path(__file__).resolve().parents[1] / "tutorials"
REPO = TUTORIALS.parents[1]


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


  def test_model_download_steps_can_run_again_in_the_same_folder(self):
    # A second tutorial in the same folder repeats the steps: without these flags
    # sima-cli, unzip and Expand-Archive stop on the files the first one left.
    pages = [*TUTORIALS.glob("[0-9][0-9][0-9]_*/README.md"),
             *TUTORIALS.glob("[0-9][0-9][0-9]_*/i18n/*/README.md")]
    for page in ("develop-apps/hello-neat/pcie.md", "develop-apps/tutorials/before-you-run.md"):
      pages += [REPO / "docs" / page, *REPO.glob(
          f"website/i18n/*/docusaurus-plugin-content-docs/current/{page}")]
    steps = 0
    for page in sorted(pages):
      for line in page.read_text(encoding="utf-8").splitlines():
        with self.subTest(page=str(page.relative_to(REPO)), line=line):
          if line.startswith("sima-cli models download "):
            self.assertIn(" --force", line)
          elif line.startswith("unzip "):
            self.assertTrue(line.startswith("unzip -o "))
          elif line.startswith("Expand-Archive "):
            self.assertTrue(line.endswith(" -Force"))
          else:
            continue
          steps += 1
    self.assertGreaterEqual(steps, 100)


if __name__ == "__main__":
  unittest.main()
