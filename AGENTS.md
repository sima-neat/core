# Core contributor instructions

Start with [CONTRIBUTING.md](CONTRIBUTING.md), the [coding standard](docs/develop-apps/contribute/start-here/coding_standard.md), and the [test requirements](docs/develop-apps/contribute/build-test/test_requirements.md). Branch from and target `develop`. Use the [PR template](.github/PULL_REQUEST_TEMPLATE.md) and link the issue.

## Choose a skill

| Task | Skill |
| --- | --- |
| Fix a runtime, binding, or tooling defect and add its regression | [core-fix-bug](.agents/skills/core-fix-bug/SKILL.md) |
| Add a capability to Core | [core-add-feature](.agents/skills/core-add-feature/SKILL.md) |
| Edit docs, examples, or tutorial guidance | [core-update-docs](.agents/skills/core-update-docs/SKILL.md) |
| Change tests, fixtures, or CI test selection | [core-update-tests](.agents/skills/core-update-tests/SKILL.md) |
| Review a Core PR or branch, including your own before requesting review | [core-review-pr](.agents/skills/core-review-pr/SKILL.md) |

Use the canonical skill file if your agent does not list it. A runtime fix and
its regression belong together under `core-fix-bug`. CI infrastructure changes
such as publishing or updating runner images follow CONTRIBUTING.md.

## Core contracts

- Keep the C++20 `simaai::neat` public API under `include/` stable. Examples and tutorials use installed APIs. For breaking changes, follow the coding standard and the ABI rule in the [release checklist](docs/develop-apps/contribute/release-maintenance/release-checklist.md).
- Core owns `Model`, `Graph`, `Run`, model and graph policy, and PyNeat. Scheduler owns the installed runtime boundary and GStreamer transport. Core consumes Scheduler and LLiMa packages. Read the [runtime contracts](.agents/runtime-contracts.md) for dependency, transport, or numerical changes.
- Use only `mpk.json` or `*_mpk.json` for model contracts. Preproc accepts user-defined input geometry and produces the model's required input shape. Follow the [MPK contract](docs/develop-apps/contribute/contracts-internals/mpk_contract.md).
- Keep graph construction deterministic and runtime waits bounded. Preserve structured diagnostics and report the failing API, input, or node. Use the [pipeline diagnostics tutorial](tutorials/012_diagnose_a_pipeline/README.md) when investigating a failure.
- Use `build.sh` with matching dependencies. Run AArch64 device tests through `dk` or the repo's DevKit wrappers. Follow the [hardware test guidance](docs/develop-apps/contribute/build-test/hardware-test-safety.md).

## Code Review Rules

- Trace the change through supported C++ and Python callers and the owning component. Check the reachable inputs, default behavior, buffer lifetime, and failure paths.
- Require a concrete case for each new option, helper, guard, or fallback. Reuse an existing contract or validation where possible. Prefer short comments that capture constraints and docs that help readers take an action.
- For a bug fix, identify the original failure and the regression that passes. For a CI selection change, check the named test passed in the intended job. Report the tested target and package or the outstanding device check.
- Check API and ABI separately. Compare `deps/manifest.json` with `main` before asking for an ABI bump. Check required translations and source hashes when localized docs change.

Use [core-review-pr](.agents/skills/core-review-pr/SKILL.md) for the full review. Give
each finding a location, triggering case, and consequence. Keep optional cleanup
separate from defects.
