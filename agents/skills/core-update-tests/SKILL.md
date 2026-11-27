---
name: core-update-tests
description: Add, change, or retire Core tests, replace model or media fixtures, or change CI test selection. Use core-fix-bug when a runtime fix and its regression are the same task.
---

# Update Core tests

## Workflow

1. **Find the lane.** Identify the behavior the test protects and its CTest
   name or pytest node ID. Trace the CMake registration and labels, or the
   pytest file and markers, to the job's selector in the owning workflow, such as
   [Vulcan CI](../../../.github/workflows/vulcan-ci.yml) or
   [Long Tests Weekly](../../../.github/workflows/long-tests-weekly.yml).
   Check the job's inclusions and exclusions. Build Tooling Tests lists pytest
   files explicitly, so update that list when adding a file to the job.
2. **Prepare fixtures.** Trace the selected job to its existing fixture helper
   or preparation step. For a Model Browser fixture used by a partitioned
   DevKit E2E test, reuse or extend the fixture mapping in the
   [E2E partitioner](../../../tools/ci/partition_devkit_e2e.py). It prepares
   archives for selected tests and passes their resolved paths to CTest.
   Prepare model and media fixtures only in jobs that need them. Pass the
   resolved path to the test. Keep downloads, cache lookup, and locking in
   fixture preparation.
   Preserve the assertion when changing a fixture. An archive pinned for a
   known bug must still let the regression pass with the fix. On retirement,
   remove the fixture mapping only if no other test uses it.
3. **Run the matching profile.** Use the profile affected by the change.
   Enable `run_long_tests` for changes to a long-only Modalix case.
   Use the repo's DevKit wrappers for AArch64 binaries. Keep required tests
   strict so missing fixtures or coverage fail their lane. Follow the
   [test requirements](../../../docs/develop-apps/contribute/build-test/test_requirements.md).
4. **Verify and report.**
   - When removing a test from a job, confirm the job no longer selects it
     and the remaining tests pass.
   - For a test the job should run, confirm it ran and passed in the job's
     coverage artifacts or log. For CTest JUnit results, check that it ran
     and has no `<skipped/>`, `<failure/>`, or `<error/>`. For pytest coverage
     results, check setup, call, and teardown. If the log gives only aggregate
     results, report a focused run in the same profile and note any outstanding
     CI verification.

   Use the existing lane coverage check to confirm selection matches execution.
   Link the result and name the tested profile, target, and any fixture in the PR.
