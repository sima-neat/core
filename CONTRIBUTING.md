# Contributing to SiMa.ai Neat Core

## Start with the issue

Use the [bug report](.github/ISSUE_TEMPLATE/bug_report.md) or [feature request](.github/ISSUE_TEMPLATE/feature_request.md)
template. Give a minimal failing case for a bug or the caller's use case for a
feature. Redact credentials and customer data from inputs and logs.

Create a topic branch from current `develop` and target `develop` in the PR.
Use focused commits with imperative subjects. Link the issue with `Fixes #<issue>`
when the change resolves it, or `Refs #<issue>` for partial work.

## Make the change

Use a checkout inside the Ubuntu Palette SDK container for development and
agent skills. Start with the [build guide](docs/develop-apps/contribute/build-test/build.md)
and the [coding standard](docs/develop-apps/contribute/start-here/coding_standard.md).
Core uses C++20, the `simaai::neat` namespace, and stable installed headers under
`include/`. Keep include groups in the existing order.

- Choose the simplest change that covers supported behavior. Reuse an existing
  API, validator, or helper when it fits. If the decode type determines the output
  format, derive it from the type and avoid a second setting.
- Add guards and fallbacks for reachable failures. Check which callers can
  bypass input validation, including PyNeat and supported older archives.
  Preserve handling of real failures such as timeouts and teardown.
- Keep comments short. Explain a constraint such as why a buffer must live through
  a completion callback. Put investigation history and design tradeoffs in the PR.
- Update existing docs when a reader needs new guidance. Follow the
  [documentation policy](docs/develop-apps/contribute/start-here/coding_standard.md#documentation-obligations)
  and update examples when API usage changes. `No docs update needed` is valid
  when a fix restores documented behavior.

Follow the coding standard for maintainer approval of breaking changes. When
feasible, keep a deprecated API available for at least one release and give
callers migration guidance. Check `abi-version` against `main` using the
[release checklist](docs/develop-apps/contribute/release-maintenance/release-checklist.md)
and describe the migration in the PR.

## Validate the affected behavior

Use the [test requirements](docs/develop-apps/contribute/build-test/test_requirements.md)
to select the checks. Use `build.sh` to prepare Core and its matching dependencies.

| Change | Evidence to include |
| --- | --- |
| Runtime bug | The failing case, the regression that now passes, and the target/package for device behavior |
| Public API or PyNeat | Compile coverage for supported C++ usage and relevant `python/tests` coverage for Python behavior |
| Test or fixture | The affected test passes on the intended target with the chosen fixture |
| CI test selection | The named test executed and passed in that job's result, including any applicable DevKit lane |
| Model archive handling | Contract tests and the [model archive validation gate](scripts/ci/run_model_archive_security_gate.sh) |
| Install or performance | The applicable [install smoke](scripts/ci/run_install_smoke.sh) or [performance gate](scripts/ci/run_perf_regression_gate.sh) result |
| Published documentation | The applicable localization checks, docs build, and strict link check |

Report commands and results, and name any checks that remain unverified. A test
that was selected but skipped is unverified. Fix crashes, use-after-free, and
data races on the current branch. A failed
[crash/correctness gate](scripts/ci/run_crash_correctness_gate.sh) blocks push or merge.

### Match the CI formatter

Code Hygiene uses the preinstalled `clang-format-18` from the pinned Ubuntu 24.04
image. Find the exact version in the job's **Verify preinstalled code hygiene tools**
log and compare it with `clang-format --version` in the SDK container. The Ubuntu
24.04 `clang-format` package provides the distro formatter. Install it with
`sudo apt-get install clang-format` if needed. If a different installation
takes precedence, put `/usr/bin` first on `PATH` for the format check.

Run the checks on the branch diff:

```bash
git fetch origin develop
bash scripts/check_format.sh --changed-only --base-ref origin/develop
bash scripts/check_cmake_format.sh --changed-only --base-ref origin/develop
bash scripts/check_duplicate_includes.sh --changed-only --base-ref origin/develop
```

Optionally install the [local Git hooks](scripts/dev/install_hooks.sh) in your
checkout to check staged files and run the pre-push crash/correctness gate.

### Update translations

For changes covered by the [localization guide](website/i18n/README.md), update
the affected translations in every supported locale in the same PR. AI can
help draft translations. Verify them against the updated source before updating
their source hashes. For a generated tutorial, update the tutorial-owned
translations and hashes at the source. Run
`bash scripts/ci/check_documentation_localization.sh` and the relevant docs build
and [link check](scripts/ci/check_docs_links.sh).

## Prepare the PR

Fill in the [PR template](.github/PULL_REQUEST_TEMPLATE.md) with the issue,
reason for the change, risk, validation, and any API, ABI, migration, or docs
impact. After implementation, tests, and docs edits, review the full diff against
`develop` using [core-review-pr](.agents/skills/core-review-pr/SKILL.md). You can
follow the skill manually. Address applicable findings, check the self-review box,
and add one sentence describing what changed during review, or write `No findings`.
Repeat after a substantial rewrite. Resolve reviewer findings and check the
required CI results before merging.

[AGENTS.md](AGENTS.md) maps tasks to the five Core skills. Codex discovers
`.agents/skills/`. Claude Code uses the committed links under `.claude/skills/`
and imports AGENTS.md through CLAUDE.md.

## Before releasing

Follow the [release checklist](docs/develop-apps/contribute/release-maintenance/release-checklist.md).
Before promoting `develop` to `main`, validate the production docs inputs on the
Core commit you will promote. Use the staging workflow with the sibling repos'
`main` branches:

```bash
gh workflow run "Vulcan Docs" --repo sima-neat/core --ref develop -f sibling_branch=main
```

Check that the promoted Core tree and the sibling SHAs in the `[autodoc]`
log lines match the versions tested in staging.
