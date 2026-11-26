---
name: core-fix-bug
description: Reproduce and fix a Core defect through the affected C++ or PyNeat API, build or install command, or tutorial. Use for behavior fixes and their regression tests.
---

# Fix a Core bug

1. Record the Core commit or package, Palette SDK version, and DevKit or host.
   Start with the reported API call, command, or tutorial when available.
   Reduce it while preserving the reported failure. Record what ran and the
   expected and actual result. For a runtime failure, capture the relevant
   structured Neat error or `GraphReport`. If the failure does not reproduce,
   record what you tried, any relevant environment mismatch, and the specific
   evidence needed next.
2. Follow the failing path through C++ and PyNeat where both can reach it.
   Identify which component owns the failure using the
   [runtime contracts](../../../agents/runtime-contracts.md). For model routing,
   read only the MPK manifest and trace the input through the failing stage.
   If the cause is unclear, use existing [pipeline diagnostics](../../../tutorials/012_diagnose_a_pipeline/README.md)
   or a focused trace to find where the value or state first goes wrong. When a
   working case is available, compare it by changing one factor at a time.
   Confirm the cause before changing shared behavior.
3. Fix it at the owning boundary. Use an existing validator or contract if it
   covers the admitted inputs. Add a guard when a supported caller can reach a
   failing state. Keep the code change focused on the cause and include the
   regression in the same patch.
4. Add a regression at the closest layer that exercises the reported behavior.
   Show the failing result before the fix and the passing result after it.
   Remove temporary probes and rerun the original reproducer when available.
   Use `python/tests/` for Python behavior and Core's CTest lanes for C++.
   Use the [test requirements](../../../docs/develop-apps/contribute/build-test/test_requirements.md)
   and DevKit wrappers for cases that depend on hardware.
5. In the PR, describe the cause, the supported behavior restored, and the
   command, result, and package/target tested. Identify any device check still
   outstanding. Update existing docs when the fix changes what readers need to
   know or do. Write `No docs update needed` when existing guidance still applies.
   For localized docs, update required translations and source hashes using the
   [localization guide](../../../website/i18n/README.md).

For a two-input `TensorSet` with an invalid attribute key, keep both inputs and
submit through the affected Model route. Check for `misconfig.sample_attributes`
before queueing. That route dropped the attributes before validation.
[Core #1497](https://github.com/sima-neat/core/pull/1497)
