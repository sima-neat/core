---
name: core-fix-bug
description: Reproduce and fix a Core defect through the affected C++ or PyNeat API, build or install command, or tutorial. Use for behavior fixes and their regression tests.
---

# Fix a Core bug

1. Record the Core commit or package, Palette SDK version, and DevKit or host.
   Reproduce with the smallest supported API call, command, or tutorial that
   shows the failure. Record the expected and actual result. For a runtime
   failure, capture the relevant structured Neat error or `GraphReport`.
2. Follow the failing path through C++ and PyNeat where both can reach it.
   Identify which component owns the failure using the
   [runtime contracts](../../../.agents/runtime-contracts.md). For model routing,
   read only the MPK manifest and trace the input through the failing stage.
   Confirm the cause before changing shared behavior.
3. Fix it at the owning boundary. Use an existing validator or contract if it
   covers the admitted inputs. Add a guard when a supported caller can reach a
   failing state. Keep the fix and its regression in the same change.
4. Add a regression at the closest layer that exercises the reported behavior.
   Show the failing result before the fix and the passing result after it. Use
   `python/tests/` for Python behavior and Core's CTest lanes for C++.
   Use the [test requirements](../../../docs/develop-apps/contribute/build-test/test_requirements.md)
   and DevKit wrappers for cases that depend on hardware.
5. In the PR, describe the cause, the supported behavior restored, and the
   command, result, and package/target tested. Identify any device check still
   outstanding. Update existing guidance if behavior now differs from it.
