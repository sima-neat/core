---
name: core-update-tests
description: Add, change, or retire Core tests, replace model or media fixtures, or change CI test selection. Use core-fix-bug when a runtime fix and its regression are the same task.
---

# Update Core tests

1. Identify the behavior the test protects, its CTest name or pytest node ID,
   and the host, DevKit shard, or weekly job that owns it. Use the
   [test requirements](../../../docs/develop-apps/contribute/build-test/test_requirements.md).
   Preserve the assertion on that behavior when changing a model or fixture.
2. Use an existing fixture helper or the preparation step for that lane in
   [Vulcan CI](../../../.github/workflows/vulcan-ci.yml). Prepare model and media
   fixtures only in jobs that need them. Pass the resolved path to the test.
   Keep downloads, cache lookup, and locking in fixture preparation. An archive
   pinned to exercise a known bug must still let the regression pass with the fix.
3. Keep required tests strict. Missing fixtures and missing coverage should fail
   their lane. Use the repo's DevKit wrappers for AArch64 binaries and keep the
   appropriate capability profile.
4. Check the intended test result in the job artifact. For CTest, confirm the
   case ran and passed in JUnit, including no `<skipped/>`, `<failure/>`, or
   `<error/>` for that case. For pytest, check the node's setup, call, and
   teardown outcomes. Use the existing lane coverage check to confirm selection
   matches execution. Link the run and name the tested target and fixture in the PR.
