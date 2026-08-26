from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTIFICATION_WORKFLOW = ROOT / ".github" / "workflows" / "vulcan-ci-notify.yml"
VULCAN_WORKFLOW = ROOT / ".github" / "workflows" / "vulcan-ci.yml"


def notification_workflow() -> str:
    return NOTIFICATION_WORKFLOW.read_text(encoding="utf-8")


def test_listener_observes_only_core_vulcan_ci_protected_branches() -> None:
    content = notification_workflow()

    assert "workflow_run:" in content
    assert "- Vulcan CI" in content
    assert "- completed" in content
    assert "- main" in content
    assert "- develop" in content


def test_listener_alerts_only_on_actionable_conclusions() -> None:
    content = notification_workflow()

    for conclusion in (
        "failure",
        "timed_out",
        "action_required",
        "startup_failure",
        "stale",
    ):
        assert f"conclusion == '{conclusion}'" in content

    assert "conclusion == 'cancelled'" not in content
    assert "conclusion == 'success'" not in content


def test_listener_uses_shared_notifier_with_minimal_permissions_and_secrets() -> None:
    content = notification_workflow()

    assert "actions: read" in content
    assert "contents: read" in content
    assert "sima-neat/.github/.github/workflows/vulcan-notify-slack.yml@main" in content
    assert "channel_id: ${{ vars.SLACK_VULCAN_EVENT_CHANNEL_ID }}" in content
    assert "slack_bot_token: ${{ secrets.SLACK_BOT_TOKEN }}" in content
    assert "secrets: inherit" not in content


def test_listener_forwards_run_diagnostics() -> None:
    content = notification_workflow()

    for field in (
        "id",
        "repository.full_name",
        "name",
        "head_branch",
        "head_sha",
        "conclusion",
        "run_attempt",
        "html_url",
        "actor.login",
        "head_commit.message",
    ):
        assert f"github.event.workflow_run.{field}" in content


def test_core_ci_runs_notification_contract_test() -> None:
    content = VULCAN_WORKFLOW.read_text(encoding="utf-8")

    assert "tests/tools/test_vulcan_ci_notification.py" in content
