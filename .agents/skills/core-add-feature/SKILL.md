---
name: core-add-feature
description: Add a user capability to Neat Core, including public C++ or PyNeat behavior, nodes, model execution, or developer tooling. Use for feature work with compatibility and user-facing validation.
---

# Add a Core feature

1. Describe the caller's workflow, desired result, and supported inputs. Look for
   an existing `Model`, `Graph`, `Run`, Node or NodeGroup composition that meets
   the need. Add a public option when the caller needs a choice that the current
   arguments and contracts cannot express.
2. Keep the public additions small and preserve existing defaults. Carry the
   behavior through affected C++ entry points and PyNeat bindings. Follow the
   [coding standard](../../../docs/develop-apps/contribute/start-here/coding_standard.md)
   and check compatibility with supported callers and packages. Apply the ABI
   rule in the [release checklist](../../../docs/develop-apps/contribute/release-maintenance/release-checklist.md).
3. Respect [runtime ownership](../../../.agents/runtime-contracts.md). Keep policy
   and MPK interpretation in Core and use the installed Scheduler interface for
   execution. For a node, exercise deterministic fragments, parse/caps validation,
   and a realistic integration chain. Match the remaining checks to the
   [test requirements](../../../docs/develop-apps/contribute/build-test/test_requirements.md).
4. Verify a realistic use case and the relevant unsupported-input path. Show the
   observable result, tested target and package, and any limitations in the PR.
   Update the existing guide or example where callers need new instructions.
