# Core issue triage

Core uses one primary category per issue: `bug`, `feature`, `enhancement`,
`documentation`, or `ci`. `needs-repro` marks missing information that blocks
investigation and is cleared once enough evidence is available.

The [triage skill](skills/core-triage/SKILL.md) defines classification and
priority. The bot can assign an empty Priority only in the initial issue-creation
run. Later runs recommend a priority in their comment. High can be assigned in any
category with concrete evidence and high confidence. Urgent assignments require
a maintainer.

Initial High assignments and evidenced Urgent recommendations alert Slack with
only the repository, issue link, and priority action. Set the organization Actions
variable `SLACK_ISSUE_TRIAGE_CHANNEL_ID` with access for Core, make `SLACK_BOT_TOKEN`
available to Core, and invite that bot to the channel. Dry runs, edits, reopens,
and reruns do not send alerts.

## Label migration

Once this label policy is active on the default branch, run **Migrate Issue Labels** with `dry_run` enabled. Review the artifact's issues
and proposed categories. Supply any ambiguous choices in `assignments`, such as
`{"123": "ci", "456": "documentation"}`, and preview again. Run with `dry_run`
disabled when all affected issues and PRs have a reviewed category.

The migration records previous release/closure annotations in comments, replaces
deprecated labels, and removes their definitions after verifying they are unused.
Existing Priority values are maintained.

Issue-event workflows and normal triage configuration load from the repository's
default branch. Changes merged to `develop` take effect when promoted there.
