# CI and docs failure notifications

`vulcan-ci-notify.yml` listens for completed Vulcan CI and Vulcan Docs runs on
`main`, `develop`, and `v*.*.*` release branches. The default-branch listener
posts actionable failures to `SLACK_VULCAN_EVENT_CHANNEL_ID` using `SLACK_BOT_TOKEN`.

The shared notifier on `main` posts the parent alert immediately, collects logs for
the exact failed attempt, and requests a tool-free API diagnosis on a bridge
runner. A hosted job appends the validated diagnosis to the parent Slack thread.
The report identifies observed errors, likely cause, confidence, and next action.
It is an automated hypothesis, not a test result or an automatic code fix.

If analysis fails, the thread reports that diagnosis is unavailable. If no bridge
runner is online, diagnosis may queue; inspect the notification workflow while
the original alert remains available. Evidence and analysis artifacts expire
after seven days. Rerunning the notification workflow creates another alert;
rerun the source build when retrying a CI/docs failure.

Core follows the shared workflow at `@main`. The shared collector resolves its
protected `main` revision and uses that SHA for analysis; Core cannot select an
implementation branch or tag. `OPENAI_API_KEY` is passed only to analysis for a
tool-free Responses API request. This uses API billing, not bridge Codex login
credentials. The model has no command or filesystem tools. Runner prerequisites and the no-Slack smoke check are documented
in `sima-neat/.github/.github/workflows/vulcan/README.md`.
