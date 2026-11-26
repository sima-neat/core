---
name: core-triage
description: Classify Core GitHub issues, identify missing evidence, and recommend priorities using the supplied issue context.
---

# Core issue triage

Core owns the public C++ and PyNeat APIs, Model/Graph/Run behavior, MPK contract
interpretation, diagnostics, packaging, tutorials, documentation, and tests.
Use the issue body and discussion together. Compare related reports in the
available context before suggesting overlap. Flag a likely duplicate for
maintainer review and explain the evidence.

## Classification

Choose one primary label and use the same value for `category`.

| Label | When to use it |
| --- | --- |
| `bug` | Existing behavior fails or produces incorrect results |
| `feature` | A new user capability is requested |
| `enhancement` | Existing functionality needs improvement or simplification |
| `documentation` | Guides, tutorials, or reference material need updating |
| `ci` | Build, test, or publishing automation needs work |

Classify by the requested outcome. Fixture migrations and test-selection work
use `ci`, better diagnostics use `enhancement`, and documentation cleanup uses
`documentation`. A broken runtime test revealing a product defect uses `bug`.
An unlabeled request for a new capability still uses `feature`.

Use `unknown` if essential context is missing. Add `needs-repro` only when a
missing command, minimal example, version, or error output materially blocks
investigation. Omit it once the discussion supplies enough information.

## Actionability

An actionable issue can have empty template sections. Ask one focused question
only when its answer would change the investigation or priority. For a bug,
look for the affected version and environment, failing command or API, expected
and actual results, and useful evidence. For a feature, look for the use case,
current limitation or workaround, and desired observable outcome.

Use Core/SDK versions and host/DevKit details relevant to the report. Model
issues may need a public model reference and MPK contract details. Python issues
may need the interpreter version, install mode, and traceback. Documentation
reports may need only a page link and the inaccurate instruction.

## Priority

Recommend a value and one sentence explaining impact and workarounds. Leave
`priority` empty when a decisive fact is missing and ask for that fact.

| Value | Meaning |
| --- | --- |
| Urgent | A critical failure needs immediate action, such as a broken release with no workaround |
| High | A supported workflow is blocked, silently produces incorrect results, or needs a new capability for a concrete delivery requirement with no practical workaround |
| Medium | Meaningful impact with limited scope or a practical workaround |
| Low | Minor friction or polish with limited impact on completing the task |

Urgent requires a maintainer's confirmation and assignment. High can be assigned
automatically in any category. Recommend High or Urgent only when reproducible
steps or equivalent concrete evidence establish the impact with high confidence.
Cite that evidence in `priority_evidence`. Set `needs_human_review` for Urgent.
An unsupported claim of “release blocker” leaves Priority empty and requires a
focused question or maintainer review. Set `needs_human_review` when judgment
or missing evidence requires a maintainer. The runner appends `priority_reason`
to the public comment and assigns Priority only on the initial issue-creation
run. Initial High assignments and Urgent recommendations trigger a Slack alert
with the issue link and priority action. Later runs support a human decision
without another alert.

## Routing and comment

Set `area` to the affected surface, such as `cpp-api`, `python-api`,
`model-contract`, `pipeline-runtime`, `diagnostics`, `docs`, `tutorials`,
`build-ci`, `packaging`, `devkit-runtime`, or `genai`.

Request extended analysis only when specific commands, package/model references,
or logs make checking a repository in `cross_reference_repos` useful. Use that
configuration's allowlist. Private implementation or customer-specific assets
require maintainer review. Flag potential vulnerabilities or sensitive reports
for a maintainer and keep the public response conservative. Ask for redacted
output or public/synthetic reproductions.

Keep the comment short. State the likely area and whether the issue is actionable,
mention concrete evidence, and ask the one question needed to move forward.
Ground any root-cause claim in the supplied evidence or inspected source.
