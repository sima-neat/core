#!/usr/bin/env python3
"""Preview and apply Core's reviewed issue-label migration."""

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def api_request(method, path, token, payload=None):
    request = urllib.request.Request(
        "https://api.github.com" + path,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        method=method,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json", "X-GitHub-Api-Version": "2022-11-28"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            content = response.read()
            return json.loads(content) if content else None
    except urllib.error.HTTPError as error:
        raise SystemExit(f"GitHub API {method} {path} failed with HTTP {error.code}") from error


def api_paginated(path, token):
    separator = "&" if "?" in path else "?"
    items = []
    page = 1
    while True:
        batch = api_request("GET", f"{path}{separator}per_page=100&page={page}", token)
        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return value


def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def migrate_labels(args: argparse.Namespace) -> None:
    token = require_env("GITHUB_TOKEN")
    repo = require_env("GITHUB_REPOSITORY")
    config = read_json(Path(args.config))
    categories = set(config["labels"]["categories"])
    deprecated = set(config["labels"]["deprecated"])
    assignments = read_json(Path(args.assignments)) if args.assignments else {}
    issues = api_paginated(f"/repos/{repo}/issues?state=all", token)
    plan = []
    for issue in issues:
        labels = {label["name"] for label in issue["labels"]}
        primary = labels & categories
        if not labels & deprecated and len(primary) <= 1:
            continue
        category = assignments.get(str(issue["number"]))
        if category is None and len(primary) == 1:
            category = next(iter(primary))
        if category is not None and category not in categories:
            raise SystemExit(f"Invalid migration category for #{issue['number']}: {category!r}")
        plan.append({
            "number": issue["number"], "title": issue["title"],
            "category": category, "remove": sorted((labels & (deprecated | categories)) - {category}),
            "preserve": sorted(labels & {"patch-candidate", "duplicate", "invalid", "wontfix"}),
        })
    print(json.dumps({"dry_run": args.dry_run == "true", "issues": plan}, indent=2))
    if args.dry_run == "true":
        return
    unresolved = [item["number"] for item in plan if item["category"] is None]
    if unresolved:
        raise SystemExit(f"Supply --assignments with a category for these issues/PRs: {unresolved}")
    marker = "<!-- sima-neat-label-migration -->"
    for item in plan:
        path = f"/repos/{repo}/issues/{item['number']}"
        if item["preserve"]:
            comments = api_paginated(f"{path}/comments", token)
            if not any(marker in (comment.get("body") or "") for comment in comments):
                names = ", ".join(f"`{label}`" for label in item["preserve"])
                api_request("POST", f"{path}/comments", token, {
                    "body": f"{marker}\nBefore label cleanup, this issue carried {names}. This records its previous release or closure annotation.",
                })
        api_request("POST", f"{path}/labels", token, {"labels": [item["category"]]})
        current = api_request("GET", path, token)
        for label in {label["name"] for label in current["labels"]} & set(item["remove"]):
            api_request("DELETE", f"{path}/labels/{urllib.parse.quote(label, safe='')}", token)
    existing = {label["name"] for label in api_paginated(f"/repos/{repo}/labels", token)}
    for label in sorted(deprecated & existing):
        encoded = urllib.parse.quote(label, safe="")
        remaining = api_request("GET", f"/repos/{repo}/issues?state=all&labels={encoded}&per_page=1", token)
        if remaining:
            raise SystemExit(f"Label {label!r} is still used; rerun the migration preview")
        api_request("DELETE", f"/repos/{repo}/labels/{encoded}", token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / ".github/issue-triage/config.json"))
    parser.add_argument("--assignments", help="JSON object mapping issue/PR numbers to reviewed categories")
    parser.add_argument("--dry-run", choices=["true", "false"], default="true")
    migrate_labels(parser.parse_args())


if __name__ == "__main__":
    main()
