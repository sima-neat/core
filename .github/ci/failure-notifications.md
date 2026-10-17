# CI and docs failure notifications

`vulcan-ci-notify.yml` listens for completed Vulcan CI and Vulcan Docs runs on
`main`, `develop`, and `v*.*.*` release branches. The default-branch listener
posts actionable failures to `SLACK_VULCAN_EVENT_CHANNEL_ID` using `SLACK_BOT_TOKEN`.

The shared notifier on `main` posts the parent alert immediately, collects logs for
the exact failed attempt, and requests a read-only Codex diagnosis on a bridge
runner. A hosted job appends the validated diagnosis to the parent Slack thread.
The report identifies observed errors, likely cause, confidence, and next action.
It is an automated hypothesis, not a test result or an automatic code fix.

If analysis fails, the thread reports that diagnosis is unavailable. If no bridge
runner is online, diagnosis may queue; inspect the notification workflow while
the original alert remains available. Evidence and analysis artifacts expire
after seven days. Rerunning the notification workflow creates another alert;
rerun the source build when retrying a CI/docs failure.

Core follows the shared repository’s `main` branch for both the reusable workflow
and `implementation_ref`, so merged shared changes are adopted without updating
Core commit pins. Runner prerequisites and the no-Slack smoke check are documented
in `sima-neat/.github/.github/workflows/vulcan/README.md`.
