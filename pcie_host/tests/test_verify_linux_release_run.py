#!/usr/bin/env python3
"""Contracts for exact GitHub Actions release-run admission."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "pciehost" / "verify_linux_release_run.py"
SPEC = importlib.util.spec_from_file_location("verify_linux_release_run_tested", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class ReleaseRunAdmissionTests(unittest.TestCase):
  repository = "sima-neat/scheduler"
  commit = "a" * 40
  run_id = 42
  run_attempt = 3
  repository_id = 1234
  # A workflow's `run-name` sets the run `name` and every job's `workflow_name`.
  run_name = "feature/windows-pcie (production)"
  workflow_path = ".github/workflows/ci.yml"
  reusable_workflow = "sima-neat/scheduler/.github/workflows/pcie-host-windows.yml"
  job_names = (
      "Build, test, and publish native Windows PCIe profile / contracts",
      "Build, test, and publish native Windows PCIe profile / Native Windows amd64",
      "Verify Scheduler build and Vulcan publication",
  )
  artifact_name = "scheduler-native-windows-amd64"

  def make_run(self) -> dict:
    return {
        "conclusion": "success",
        "display_title": self.run_name,
        "event": "workflow_dispatch",
        "head_branch": "feature/windows-pcie",
        "head_repository": {
            "full_name": self.repository,
            "id": self.repository_id,
        },
        "head_sha": self.commit,
        "html_url": "https://github.example/run/42",
        "id": self.run_id,
        "name": self.run_name,
        "path": self.workflow_path,
        "referenced_workflows": [{
            "path": f"{self.reusable_workflow}@{self.commit}",
            "ref": "refs/heads/feature/windows-pcie",
            "sha": self.commit,
        }],
        "repository": {"full_name": self.repository, "id": self.repository_id},
        "run_attempt": self.run_attempt,
        "status": "completed",
        "workflow_id": 386851362,
    }

  def make_jobs(self) -> list[dict]:
    return [
        {
            "conclusion": "success",
            "head_sha": self.commit,
            "id": 100 + index,
            "name": name,
            "run_attempt": self.run_attempt,
            "run_id": self.run_id,
            "status": "completed",
            "workflow_name": self.run_name,
        }
        for index, name in enumerate(self.job_names)
    ]

  def make_artifacts(self) -> list[dict]:
    return [{
        "digest": "sha256:" + "b" * 64,
        "expired": False,
        "id": 900,
        "name": self.artifact_name,
        "size_in_bytes": 4096,
        "workflow_run": {
            "head_repository_id": self.repository_id,
            "head_sha": self.commit,
            "id": self.run_id,
            "repository_id": self.repository_id,
        },
    }]

  def verify(self, run=None, jobs=None, artifacts=None):
    return MODULE.verify_run(
        self.make_run() if run is None else run,
        self.repository,
        self.run_id,
        self.commit,
        self.workflow_path,
        required_reusable_workflow=self.reusable_workflow,
        required_job_names=self.job_names,
        required_artifact_name=self.artifact_name,
        jobs=self.make_jobs() if jobs is None else jobs,
        artifacts=self.make_artifacts() if artifacts is None else artifacts,
    )

  def test_strict_admission_records_exact_workflow_jobs_and_artifact(self):
    receipt = self.verify()
    self.assertEqual(receipt["reusable_workflow"], {
        "path": f"{self.reusable_workflow}@{self.commit}",
        "ref": "refs/heads/feature/windows-pcie",
        "sha": self.commit,
    })
    self.assertEqual([job["name"] for job in receipt["jobs"]], list(self.job_names))
    self.assertEqual(receipt["artifact"], {
        "digest": "sha256:" + "b" * 64,
        "id": 900,
        "name": self.artifact_name,
        "size_in_bytes": 4096,
    })

  def test_generic_admission_does_not_require_strict_inventories(self):
    run = self.make_run()
    receipt = MODULE.verify_run(
        run, self.repository, self.run_id, self.commit, self.workflow_path)
    self.assertNotIn("jobs", receipt)
    self.assertNotIn("artifact", receipt)
    self.assertNotIn("reusable_workflow", receipt)

  def test_run_is_identified_by_workflow_path_not_run_name(self):
    # Run 45140904350 of sima-neat/core: run-name made `name` the branch title.
    core_run = {
        "conclusion": "success",
        "display_title": "codex/1074-windows-pcie (production)",
        "event": "workflow_dispatch",
        "head_branch": "codex/1074-windows-pcie",
        "head_repository": {"full_name": "sima-neat/core", "id": 1},
        "head_sha": "71474ed3615341af879201fd43c8ef0d10c07712",
        "id": 45140904350,
        "name": "codex/1074-windows-pcie (production)",
        "path": ".github/workflows/vulcan-ci.yml",
        "repository": {"full_name": "sima-neat/core", "id": 1},
        "run_attempt": 1,
        "status": "completed",
        "workflow_id": 280517358,
    }
    receipt = MODULE.verify_run(
        core_run, "sima-neat/core", 45140904350,
        "71474ed3615341af879201fd43c8ef0d10c07712")
    self.assertEqual(receipt["workflow"], ".github/workflows/vulcan-ci.yml")
    self.assertEqual(self.verify()["workflow"], self.workflow_path)

    for path in (".github/workflows/other.yml", None):
      with self.subTest(path=path):
        with self.assertRaisesRegex(ValueError, "unexpected path"):
          MODULE.verify_run(
              dict(core_run, path=path), "sima-neat/core", 45140904350,
              "71474ed3615341af879201fd43c8ef0d10c07712")

  def test_strict_selectors_must_be_complete(self):
    with self.assertRaisesRegex(ValueError, "requires reusable workflow, jobs, and artifact"):
      MODULE.verify_run(
          self.make_run(), self.repository, self.run_id, self.commit,
          self.workflow_path,
          required_reusable_workflow=self.reusable_workflow,
          jobs=self.make_jobs(), artifacts=self.make_artifacts())
    with self.assertRaisesRegex(ValueError, "requires job and artifact inventories"):
      MODULE.verify_run(
          self.make_run(), self.repository, self.run_id, self.commit,
          self.workflow_path,
          required_reusable_workflow=self.reusable_workflow,
          required_job_names=self.job_names,
          required_artifact_name=self.artifact_name)

  def test_strict_admission_requires_exact_source_dispatch(self):
    for event in ("pull_request", "push", None):
      with self.subTest(event=event):
        run = self.make_run()
        run["event"] = event
        with self.assertRaisesRegex(ValueError, "exact-source workflow_dispatch"):
          self.verify(run=run)

  def test_reusable_workflow_is_exactly_bound_to_branch_and_commit(self):
    mutations = {
        "missing inventory": lambda run: run.pop("referenced_workflows"),
        "missing workflow": lambda run: run.update(referenced_workflows=[]),
        "duplicate workflow": lambda run: run["referenced_workflows"].append(
            copy.deepcopy(run["referenced_workflows"][0])),
        "wrong path": lambda run: run["referenced_workflows"][0].update(
            path=f"sima-neat/scheduler/.github/workflows/other.yml@{self.commit}"),
        "wrong sha": lambda run: run["referenced_workflows"][0].update(sha="c" * 40),
        "wrong ref": lambda run: run["referenced_workflows"][0].update(
            ref="refs/heads/other"),
        "pull request merge": lambda run: run["referenced_workflows"][0].update(
            path=f"{self.reusable_workflow}@{'d' * 40}",
            ref="refs/pull/72/merge", sha="d" * 40),
    }
    for name, mutate in mutations.items():
      with self.subTest(name=name):
        run = self.make_run()
        mutate(run)
        with self.assertRaisesRegex(ValueError, "referenced workflow|reusable workflow"):
          self.verify(run=run)

  def test_required_jobs_fail_closed_on_identity_or_result_drift(self):
    field_mutations = {
        "conclusion": "failure",
        "head_sha": "c" * 40,
        "id": 0,
        "run_attempt": self.run_attempt + 1,
        "run_id": self.run_id + 1,
        "status": "in_progress",
    }
    for field, value in field_mutations.items():
      with self.subTest(field=field):
        jobs = self.make_jobs()
        jobs[0][field] = value
        with self.assertRaisesRegex(ValueError, "required job"):
          self.verify(jobs=jobs)

    for name, jobs in (
        ("missing", self.make_jobs()[1:]),
        ("duplicate", [*self.make_jobs(), copy.deepcopy(self.make_jobs()[0])]),
        ("invalid inventory", [None]),
    ):
      with self.subTest(cardinality=name):
        with self.assertRaisesRegex(ValueError, "exactly once|inventory is invalid"):
          self.verify(jobs=jobs)

  def test_required_artifact_fails_closed_on_identity_or_integrity_drift(self):
    mutations = {
        "expired": lambda artifact: artifact.update(expired=True),
        "invalid id": lambda artifact: artifact.update(id=0),
        "empty": lambda artifact: artifact.update(size_in_bytes=0),
        "oversized": lambda artifact: artifact.update(
            size_in_bytes=MODULE.MAX_ARTIFACT_SIZE_BYTES + 1),
        "bad digest": lambda artifact: artifact.update(digest="sha256:not-a-digest"),
        "missing binding": lambda artifact: artifact.pop("workflow_run"),
        "wrong run": lambda artifact: artifact["workflow_run"].update(id=self.run_id + 1),
        "wrong head": lambda artifact: artifact["workflow_run"].update(head_sha="c" * 40),
        "wrong repository": lambda artifact: artifact["workflow_run"].update(
            repository_id=self.repository_id + 1),
    }
    for name, mutate in mutations.items():
      with self.subTest(name=name):
        artifacts = self.make_artifacts()
        mutate(artifacts[0])
        with self.assertRaisesRegex(ValueError, "required artifact"):
          self.verify(artifacts=artifacts)

    for name, artifacts in (
        ("missing", []),
        ("duplicate", [*self.make_artifacts(), copy.deepcopy(self.make_artifacts()[0])]),
        ("invalid inventory", [None]),
    ):
      with self.subTest(cardinality=name):
        with self.assertRaisesRegex(ValueError, "exactly once|inventory is invalid"):
          self.verify(artifacts=artifacts)

  def test_paginated_collection_fetch_is_bounded_and_complete(self):
    first = [{"id": index} for index in range(MODULE.PAGE_SIZE)]
    second = [{"id": MODULE.PAGE_SIZE}]
    responses = [
        {"total_count": MODULE.PAGE_SIZE + 1, "jobs": first},
        {"total_count": MODULE.PAGE_SIZE + 1, "jobs": second},
    ]
    with mock.patch.object(MODULE, "fetch_json", side_effect=responses) as fetch:
      items = MODULE.fetch_collection(self.repository, self.run_id, "token", "jobs")
    self.assertEqual(len(items), MODULE.PAGE_SIZE + 1)
    self.assertIn("per_page=100&page=1", fetch.call_args_list[0].args[0])
    self.assertIn("per_page=100&page=2", fetch.call_args_list[1].args[0])

    with mock.patch.object(MODULE, "fetch_json", side_effect=[
        {"total_count": 2, "artifacts": [{"id": 1}]},
        {"total_count": 3, "artifacts": [{"id": 2}]},
    ]):
      with self.assertRaisesRegex(ValueError, "total_count changed"):
        MODULE.fetch_collection(self.repository, self.run_id, "token", "artifacts")

    for name, response, error in (
        ("empty page", {"total_count": 2, "jobs": []}, "pagination is incomplete"),
        ("invalid total", {"total_count": MODULE.MAX_COLLECTION_ITEMS + 1, "jobs": []},
         "invalid total_count"),
        ("invalid items", {"total_count": 1, "jobs": [None]}, "invalid items"),
    ):
      with self.subTest(name=name), \
          mock.patch.object(MODULE, "fetch_json", return_value=response):
        with self.assertRaisesRegex(ValueError, error):
          MODULE.fetch_collection(self.repository, self.run_id, "token", "jobs")
    with self.assertRaisesRegex(ValueError, "unsupported"):
      MODULE.fetch_collection(self.repository, self.run_id, "token", "logs")

  def test_strict_cli_fetches_and_persists_all_admission_evidence(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "receipt.json"
      arguments = [
          str(SCRIPT),
          "--repository", self.repository,
          "--run-id", str(self.run_id),
          "--expected-commit", self.commit,
          "--workflow-path", self.workflow_path,
          "--required-reusable-workflow", self.reusable_workflow,
          "--required-artifact", self.artifact_name,
          "--output", str(output),
      ]
      for job_name in self.job_names:
        arguments.extend(("--required-job", job_name))
      with mock.patch.object(sys, "argv", arguments), \
          mock.patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}), \
          mock.patch.object(MODULE, "fetch_run", return_value=self.make_run()), \
          mock.patch.object(
              MODULE, "fetch_collection", side_effect=[self.make_jobs(), self.make_artifacts()]
          ) as fetch:
        MODULE.main()
      receipt = json.loads(output.read_text(encoding="utf-8"))
      self.assertEqual(receipt["artifact"]["name"], self.artifact_name)
      self.assertEqual(
          [call.args[3] for call in fetch.call_args_list], ["jobs", "artifacts"])


if __name__ == "__main__":
  unittest.main()
