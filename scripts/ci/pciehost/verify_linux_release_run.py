#!/usr/bin/env python3
"""Verify that a required GitHub Actions workflow passed for an exact commit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import urllib.request


def verify_run(run: dict, repository: str, run_id: int, expected_commit: str,
               workflow_name: str = "Vulcan CI",
               workflow_path: str = ".github/workflows/vulcan-ci.yml") -> dict:
  expected = {
      "conclusion": "success",
      "head_sha": expected_commit,
      "id": run_id,
      "name": workflow_name,
      "path": workflow_path,
      "status": "completed",
  }
  for name, value in expected.items():
    if run.get(name) != value:
      raise ValueError(
          f"required release run has unexpected {name}: {run.get(name)!r}; expected {value!r}")
  head_repository = run.get("head_repository")
  if not isinstance(head_repository, dict) or head_repository.get("full_name") != repository:
    raise ValueError("required release run belongs to a different repository")
  return {
      "conclusion": run["conclusion"],
      "event": run.get("event"),
      "head_sha": run["head_sha"],
      "html_url": run.get("html_url"),
      "repository": repository,
      "run_attempt": run.get("run_attempt"),
      "run_id": run_id,
      "workflow": run["name"],
  }


def fetch_run(repository: str, run_id: int, token: str) -> dict:
  if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
    raise ValueError(f"invalid GitHub repository: {repository!r}")
  request = urllib.request.Request(
      f"https://api.github.com/repos/{repository}/actions/runs/{run_id}",
      headers={
          "Accept": "application/vnd.github+json",
          "Authorization": f"Bearer {token}",
          "X-GitHub-Api-Version": "2022-11-28",
      },
  )
  with urllib.request.urlopen(request, timeout=30) as response:
    value = json.load(response)
  if not isinstance(value, dict):
    raise ValueError("GitHub Actions run response is not an object")
  return value


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--expected-commit", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--repository", required=True)
  parser.add_argument("--run-id", type=int, required=True)
  parser.add_argument("--workflow-name", default="Vulcan CI")
  parser.add_argument("--workflow-path", default=".github/workflows/vulcan-ci.yml")
  args = parser.parse_args()
  if args.run_id <= 0:
    parser.error("--run-id must be positive")
  if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
    parser.error("--expected-commit must be a full lowercase Git commit")
  token = os.environ.get("GITHUB_TOKEN", "")
  if not token:
    parser.error("GITHUB_TOKEN is required")
  receipt = verify_run(
      fetch_run(args.repository, args.run_id, token),
      args.repository, args.run_id, args.expected_commit,
      args.workflow_name, args.workflow_path)
  args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
