---
name: core-review-pr
description: Review a Neat Core PR, branch, or working diff for supported behavior, C++ and PyNeat compatibility, test evidence, code simplicity, and useful docs. Use for author self-review and maintainer review.
---

# Review a Core PR

Read [AGENTS.md](../../../AGENTS.md), the linked rules relevant to the change,
and the issue or PR description. Review the full diff from the merge base of
the proposed changes and the PR's target. Use `develop` for a normal Core
contribution. Include staged and working changes for a local review and check
`git status` for relevant new files. State the base and revision reviewed.

## Follow the behavior

1. Compare the PR's claimed behavior with the linked issue and any approved
   design. Trace each claim through the affected code, including installed
   C++ headers and PyNeat entry points where relevant. Identify the test or
   check that verifies it. When the PR includes a flow sketch, check that it
   matches the code path. Check supported inputs, defaults, failure paths,
   and buffer ownership and lifetime across push, pull, and teardown.
2. Check the [runtime contracts](../../../agents/runtime-contracts.md) when the
   change crosses Scheduler, LLiMa, GStreamer transport, or an MPK boundary.
   Trace geometry, layout, dtype, and strides to the consumer. For numerical
   changes, check quantization scales, rounding, and saturation where they
   affect results. Keep diagnostics consistent through affected routes.
3. Check API and ABI separately. Compare the manifest's `abi-version` on
   `develop` and `main` before requesting a bump for an incompatible public
   layout or binary contract change. If `develop` already carries the bump for
   this release, keep it. Follow the
   [release checklist](../../../docs/develop-apps/contribute/release-maintenance/release-checklist.md)
   and check the PR's downstream migration and approval evidence where applicable.

| Change | Check together |
| --- | --- |
| Public option default | C++ and PyNeat defaults where exposed, and regression coverage of the default behavior. |
| Submission validation | `push` and `try_push` on affected paths, including Model and Graph routes. |

## Check simplicity, docs, and evidence

- Look for duplicate validation, extra flags, unnecessary abstraction, and
  fallback paths. For each proposed guard, name how a supported caller reaches
  the state. For each simplification, show the code to change and explain which
  caller behavior it preserves. Keep guards for real runtime failures.
- Keep implementation comments to at most two lines. Flag comments that
  narrate the code. Use comments to explain constraints such as ownership,
  lifetime, or ordering. Put longer design rationale in the PR.
- Apply the [documentation policy](../../../docs/develop-apps/contribute/start-here/coding_standard.md#documentation-obligations).
  Check that changed docs help a reader use the API or workflow and that required
  translations and source hashes match.
- Check the original failing case, the regression result, and the relevant
  package and device. For CI test selection, check the named test executed and
  passed in the intended job. A selection log or skipped result is insufficient.
  For evidence from an earlier revision, check whether later changes affect
  what it proves. Run focused checks when practical and state what remains
  unverified.

## Report the review

Give the merge verdict, then order findings by impact. Report one finding per
root cause. On follow-up reviews, check prior findings against the current
diff. For each defect, give the file and line, triggering case, consequence,
and expected correction. Use a question or evidence gap when a defect is
uncertain. Separate optional cleanup from merge blockers. For self-review,
address applicable findings and record a one-sentence **Review outcome** in the
[PR template](../../../.github/PULL_REQUEST_TEMPLATE.md), or write **No findings**.
