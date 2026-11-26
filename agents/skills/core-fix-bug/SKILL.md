---
name: core-fix-bug
description: Reproduce and fix a Core defect through the affected C++ or PyNeat API, build or install command, or tutorial. Use for behavior fixes and their regression tests.
---

# Fix a Core bug

## Workflow

1. **Reproduce.** Record the Core commit or package, Palette SDK version, and
   DevKit or host. Start with the reported API call, command, or tutorial when
   available. Reduce it while preserving the reported failure. Record what ran
   and the expected and actual result. For a runtime failure, capture the
   relevant structured Neat error or `GraphReport`. If the failure does not
   reproduce, record what you tried, any relevant environment mismatch, and
   the specific evidence needed next.
2. **Trace.** Follow the failing path through C++ and PyNeat where both can
   reach it. For model routing, read only the MPK manifest and trace the input
   through the failing stage.
   If the cause is unclear, use existing [pipeline diagnostics](../../../tutorials/012_diagnose_a_pipeline/README.md)
   or a focused trace to find where the value or state first goes wrong. When a
   working case is available, compare it by changing one factor at a time.
   Confirm the cause before changing shared behavior. Use the
   [runtime contracts](../../../agents/runtime-contracts.md) to identify which
   repository owns the broken contract. If the fix belongs in another repo,
   identify any Core integration work it requires.
3. **Fix.** Make the smallest change that addresses the cause at the owning
   boundary. Use an existing validator or contract if it covers the admitted
   inputs. Add a guard when a supported caller can reach a failing state.
   Include the regression in the same patch.
   If viable fixes differ in caller behavior, component ownership, or
   maintenance cost, compare them against the existing contract. If the choice
   is still open, explain the two strongest options and recommend one with a
   reason. Ask the contributor to confirm the direction with the relevant code
   owner before implementing it.
4. **Verify.** Add a regression at the closest layer that exercises the
   reported behavior. Show the failing result before the fix and the passing
   result after it. Remove temporary probes and rerun the original reproducer
   when available. Use `python/tests/` for Python behavior and Core's CTest
   lanes for C++. Use the [test requirements](../../../docs/develop-apps/contribute/build-test/test_requirements.md)
   and DevKit wrappers for cases that depend on hardware.
5. **Report and document.** Describe the cause and the supported behavior
   restored in the PR. Include the test command, result, package, target, and
   any outstanding device checks. Update existing docs when the fix changes
   what readers need to know or do. Write `No docs update needed` when existing
   guidance still applies. For docs edits, follow
   [core-update-docs](../core-update-docs/SKILL.md).
