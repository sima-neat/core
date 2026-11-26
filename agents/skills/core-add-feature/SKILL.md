---
name: core-add-feature
description: Add a user capability to Neat Core, including public C++ or PyNeat behavior, nodes, model execution, or developer tooling. Use for feature work with compatibility and user-facing validation.
---

# Add a Core feature

## Workflow

1. **Validate the need.** For an existing feature request, confirm the caller's
   goal. For a new feature idea, describe the caller's workflow, supported
   inputs, and expected result. In both cases, check whether existing `Model`,
   `Graph`, `Run`, Node, or NodeGroup APIs can meet the need. Show the existing
   solution when it fits. Before proposing a design, state the missing
   capability and which repository owns it. Use the
   [runtime contracts](../../../agents/runtime-contracts.md) to check runtime
   ownership. If another repo owns it, identify any Core integration needed.
2. **Propose the design.** Keep public additions small and preserve existing
   defaults. Derive behavior from existing inputs or options when they
   determine it. Add a public option only when callers need an independent
   choice that the current API cannot express.

   Show one realistic call or command with its input and expected result.
   Describe the recommended approach, compatibility impact, and verification
   plan. If you opened the feature issue, put the proposal in its description.
   If someone else opened the feature request, write the proposal as a comment
   on that issue. If viable designs differ in caller behavior or component
   ownership and the choice is still open, explain the two strongest options
   and recommend one with a reason.

   Get the relevant code owner's approval before implementing a feature that
   adds or changes a public API, changes a default, or shifts responsibility
   between components. Reuse an approved design already recorded in the issue.

3. **Implement at the owning boundary.** Reuse existing APIs. Keep policy
   and MPK interpretation in Core and use the installed Scheduler interface for
   execution. Carry the behavior through affected C++ entry points and
   PyNeat bindings. Follow the
   [coding standard](../../../docs/develop-apps/contribute/start-here/coding_standard.md)
   and check compatibility with supported callers and packages. Apply the ABI
   rule in the [release checklist](../../../docs/develop-apps/contribute/release-maintenance/release-checklist.md).
4. **Verify.** Exercise the new use case and the relevant unsupported-input
   path. When the feature touches shared behavior, verify the existing default.
   For a node, exercise deterministic fragments, parse/caps validation, and a
   realistic integration chain. Match the remaining checks to the
   [test requirements](../../../docs/develop-apps/contribute/build-test/test_requirements.md).
5. **Report and document.** Explain the design choice in the PR and include
   the test command, observable result, tested target and package, and any
   limitations. Update the existing guide, reference, or example where callers
   need new instructions. For localized docs, update required translations and
   source hashes using the [localization guide](../../../website/i18n/README.md).
