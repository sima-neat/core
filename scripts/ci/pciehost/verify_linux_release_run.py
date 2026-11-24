#!/usr/bin/env python3
"""Verify that a required GitHub Actions workflow passed for an exact commit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import urllib.request


GITHUB_API = "https://api.github.com"
PAGE_SIZE = 100
MAX_COLLECTION_ITEMS = 10_000
MAX_ARTIFACT_SIZE_BYTES = 2 * 1024 * 1024 * 1024


def validate_repository(repository: str) -> None:
  if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
    raise ValueError(f"invalid GitHub repository: {repository!r}")


def fetch_json(url: str, token: str) -> object:
  request = urllib.request.Request(
      url,
      headers={
          "Accept": "application/vnd.github+json",
          "Authorization": f"Bearer {token}",
          "X-GitHub-Api-Version": "2022-11-28",
      },
  )
  with urllib.request.urlopen(request, timeout=30) as response:
    return json.load(response)


def fetch_collection(repository: str, run_id: int, token: str,
                     collection: str) -> list[dict]:
  """Fetch one bounded Actions run collection without following arbitrary links."""
  validate_repository(repository)
  if run_id <= 0:
    raise ValueError("run_id must be positive")
  if collection not in ("jobs", "artifacts"):
    raise ValueError(f"unsupported Actions run collection: {collection!r}")

  items: list[dict] = []
  total_count: int | None = None
  for page in range(1, (MAX_COLLECTION_ITEMS // PAGE_SIZE) + 2):
    url = (f"{GITHUB_API}/repos/{repository}/actions/runs/{run_id}/{collection}"
           f"?per_page={PAGE_SIZE}&page={page}")
    value = fetch_json(url, token)
    if not isinstance(value, dict):
      raise ValueError(f"GitHub Actions {collection} response is not an object")
    page_total = value.get("total_count")
    page_items = value.get(collection)
    if (not isinstance(page_total, int) or isinstance(page_total, bool)
        or page_total < 0 or page_total > MAX_COLLECTION_ITEMS):
      raise ValueError(f"GitHub Actions {collection} response has invalid total_count")
    if not isinstance(page_items, list) or any(not isinstance(item, dict) for item in page_items):
      raise ValueError(f"GitHub Actions {collection} response has invalid items")
    if total_count is None:
      total_count = page_total
    elif page_total != total_count:
      raise ValueError(f"GitHub Actions {collection} total_count changed during pagination")
    items.extend(page_items)
    if len(items) == total_count:
      return items
    if len(items) > total_count or not page_items:
      raise ValueError(f"GitHub Actions {collection} pagination is incomplete")
  raise ValueError(f"GitHub Actions {collection} pagination exceeded its bound")


def _positive_integer(value: object) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value > 0


def verify_reusable_workflow(run: dict, expected_commit: str,
                             workflow_path: str) -> dict[str, str]:
  branch = run.get("head_branch")
  if not isinstance(branch, str) or not branch:
    raise ValueError("required release run has no head branch")
  expected = {
      "path": f"{workflow_path}@{expected_commit}",
      "ref": f"refs/heads/{branch}",
      "sha": expected_commit,
  }
  workflows = run.get("referenced_workflows")
  if not isinstance(workflows, list):
    raise ValueError("required release run has no referenced workflow inventory")
  matches = []
  for workflow in workflows:
    path = workflow.get("path") if isinstance(workflow, dict) else None
    if isinstance(path, str) and path.rsplit("@", 1)[0] == workflow_path:
      matches.append(workflow)
  if len(matches) != 1:
    raise ValueError(
        "required release run must reference the exact reusable workflow once")
  workflow = matches[0]
  for name, value in expected.items():
    if workflow.get(name) != value:
      raise ValueError(
          f"required reusable workflow has unexpected {name}: "
          f"{workflow.get(name)!r}; expected {value!r}")
  return expected


def verify_jobs(jobs: list[dict], run: dict, run_id: int, expected_commit: str,
                required_names: tuple[str, ...]) -> list[dict]:
  if not isinstance(jobs, list) or any(not isinstance(job, dict) for job in jobs):
    raise ValueError("required job inventory is invalid")
  if not required_names or len(required_names) != len(set(required_names)):
    raise ValueError("required job names must be a nonempty unique list")
  run_attempt = run.get("run_attempt")
  if not _positive_integer(run_attempt):
    raise ValueError("required release run has an invalid run_attempt")
  admitted = []
  for required_name in required_names:
    if not isinstance(required_name, str) or not required_name:
      raise ValueError("required job names must be nonempty strings")
    matches = [job for job in jobs if job.get("name") == required_name]
    if len(matches) != 1:
      raise ValueError(f"required release run must contain job {required_name!r} exactly once")
    job = matches[0]
    expected = {
        "conclusion": "success",
        "head_sha": expected_commit,
        "run_attempt": run_attempt,
        "run_id": run_id,
        "status": "completed",
    }
    for name, value in expected.items():
      if job.get(name) != value:
        raise ValueError(
            f"required job {required_name!r} has unexpected {name}: "
            f"{job.get(name)!r}; expected {value!r}")
    if not _positive_integer(job.get("id")):
      raise ValueError(f"required job {required_name!r} has an invalid id")
    admitted.append({
        "conclusion": job["conclusion"],
        "id": job["id"],
        "name": job["name"],
        "run_attempt": job["run_attempt"],
    })
  return admitted


def verify_artifact(artifacts: list[dict], run: dict, run_id: int,
                    expected_commit: str, required_name: str) -> dict:
  if (not isinstance(artifacts, list)
      or any(not isinstance(artifact, dict) for artifact in artifacts)):
    raise ValueError("required artifact inventory is invalid")
  if not isinstance(required_name, str) or not required_name:
    raise ValueError("required artifact name must be a nonempty string")
  matches = [artifact for artifact in artifacts if artifact.get("name") == required_name]
  if len(matches) != 1:
    raise ValueError(
        f"required release run must contain artifact {required_name!r} exactly once")
  artifact = matches[0]
  if artifact.get("expired") is not False:
    raise ValueError(f"required artifact {required_name!r} is expired")
  if not _positive_integer(artifact.get("id")):
    raise ValueError(f"required artifact {required_name!r} has an invalid id")
  if not _positive_integer(artifact.get("size_in_bytes")):
    raise ValueError(f"required artifact {required_name!r} is empty")
  if artifact["size_in_bytes"] > MAX_ARTIFACT_SIZE_BYTES:
    raise ValueError(f"required artifact {required_name!r} exceeds the size bound")
  digest = artifact.get("digest")
  if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
    raise ValueError(f"required artifact {required_name!r} has an invalid sha256 digest")

  workflow_run = artifact.get("workflow_run")
  if not isinstance(workflow_run, dict):
    raise ValueError(f"required artifact {required_name!r} has no workflow run binding")
  expected_binding = {"head_sha": expected_commit, "id": run_id}
  for name, value in expected_binding.items():
    if workflow_run.get(name) != value:
      raise ValueError(
          f"required artifact {required_name!r} has unexpected workflow_run.{name}: "
          f"{workflow_run.get(name)!r}; expected {value!r}")

  repository = run.get("repository")
  head_repository = run.get("head_repository")
  repository_id = repository.get("id") if isinstance(repository, dict) else None
  head_repository_id = (
      head_repository.get("id") if isinstance(head_repository, dict) else None)
  if (not _positive_integer(repository_id) or repository_id != head_repository_id
      or workflow_run.get("repository_id") != repository_id
      or workflow_run.get("head_repository_id") != head_repository_id):
    raise ValueError(f"required artifact {required_name!r} has a different repository binding")
  return {
      "digest": digest,
      "id": artifact["id"],
      "name": required_name,
      "size_in_bytes": artifact["size_in_bytes"],
  }


def verify_run(run: dict, repository: str, run_id: int, expected_commit: str,
               workflow_path: str = ".github/workflows/vulcan-ci.yml", *,
               required_reusable_workflow: str | None = None,
               required_job_names: tuple[str, ...] = (),
               required_artifact_name: str | None = None,
               jobs: list[dict] | None = None,
               artifacts: list[dict] | None = None) -> dict:
  # The workflow file identifies the run: run-name replaces `name` with a per-run title.
  expected = {
      "conclusion": "success",
      "head_sha": expected_commit,
      "id": run_id,
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
  receipt = {
      "conclusion": run["conclusion"],
      "event": run.get("event"),
      "head_sha": run["head_sha"],
      "html_url": run.get("html_url"),
      "repository": repository,
      "run_attempt": run.get("run_attempt"),
      "run_id": run_id,
      "workflow": run["path"],
  }
  strict = bool(
      required_reusable_workflow or required_job_names or required_artifact_name)
  if strict:
    if run.get("event") != "workflow_dispatch":
      raise ValueError(
          "strict run admission requires an exact-source workflow_dispatch run")
    if not (required_reusable_workflow and required_job_names and required_artifact_name):
      raise ValueError(
          "strict run admission requires reusable workflow, jobs, and artifact selectors")
    if jobs is None or artifacts is None:
      raise ValueError("strict run admission requires job and artifact inventories")
    receipt["reusable_workflow"] = verify_reusable_workflow(
        run, expected_commit, required_reusable_workflow)
    receipt["jobs"] = verify_jobs(
        jobs, run, run_id, expected_commit, required_job_names)
    receipt["artifact"] = verify_artifact(
        artifacts, run, run_id, expected_commit, required_artifact_name)
  return receipt


def fetch_run(repository: str, run_id: int, token: str) -> dict:
  validate_repository(repository)
  if run_id <= 0:
    raise ValueError("run_id must be positive")
  value = fetch_json(
      f"{GITHUB_API}/repos/{repository}/actions/runs/{run_id}", token)
  if not isinstance(value, dict):
    raise ValueError("GitHub Actions run response is not an object")
  return value


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--expected-commit", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--repository", required=True)
  parser.add_argument("--run-id", type=int, required=True)
  parser.add_argument("--workflow-path", default=".github/workflows/vulcan-ci.yml")
  parser.add_argument("--required-reusable-workflow")
  parser.add_argument("--required-job", action="append", default=[])
  parser.add_argument("--required-artifact")
  args = parser.parse_args()
  if args.run_id <= 0:
    parser.error("--run-id must be positive")
  if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
    parser.error("--expected-commit must be a full lowercase Git commit")
  token = os.environ.get("GITHUB_TOKEN", "")
  if not token:
    parser.error("GITHUB_TOKEN is required")
  run = fetch_run(args.repository, args.run_id, token)
  strict = bool(
      args.required_reusable_workflow or args.required_job or args.required_artifact)
  jobs = fetch_collection(args.repository, args.run_id, token, "jobs") if strict else None
  artifacts = (
      fetch_collection(args.repository, args.run_id, token, "artifacts") if strict else None)
  receipt = verify_run(
      run,
      args.repository, args.run_id, args.expected_commit,
      args.workflow_path,
      required_reusable_workflow=args.required_reusable_workflow,
      required_job_names=tuple(args.required_job),
      required_artifact_name=args.required_artifact,
      jobs=jobs,
      artifacts=artifacts)
  args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
