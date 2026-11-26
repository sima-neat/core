#!/usr/bin/env python3
"""Check the contributor skill layout in a Core checkout."""

import subprocess
from pathlib import Path


SKILLS = (
    "core-fix-bug",
    "core-add-feature",
    "core-update-docs",
    "core-update-tests",
    "core-review-pr",
)


def main():
    root = Path(__file__).resolve().parents[2]
    if (root / "CLAUDE.md").read_bytes() != b"@AGENTS.md\n":
        raise SystemExit("CLAUDE.md must contain only the @AGENTS.md import")

    for name in SKILLS:
        path = Path(".claude/skills", name)
        link = root / path
        entry = subprocess.check_output(
            ["git", "ls-files", "--stage", "--", path], cwd=root, text=True
        )
        if not entry.startswith("120000 ") or not link.is_symlink():
            raise SystemExit(f"{path}: expected a tracked Git symlink (mode 120000)")
        target = Path("../../.agents/skills", name)
        if link.readlink() != target:
            raise SystemExit(f"{path}: expected the relative target {target}")
        skill = link / "SKILL.md"
        if not skill.is_file() or not skill.read_text(encoding="utf-8").strip():
            raise SystemExit(f"{path}/SKILL.md: expected a readable contributor skill")

    print(f"Checked CLAUDE.md and {len(SKILLS)} contributor skill links")


if __name__ == "__main__":
    main()
