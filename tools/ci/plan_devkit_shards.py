#!/usr/bin/env python3
"""Plan up to six DevKit workers, preserving all canonical Core test lanes."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

# Seconds from successful develop run 40626198772. Setup is per worker.
LANES = {"fast": 230, "e2e-0": 257, "e2e-1": 257,
         "genai": 129, "performance": 276, "python-tutorials": 410}

def collect(get, path, key):
    result = []
    for page in range(1, 1001):
        data = get(f"{path}?per_page=100&page={page}")[key]
        if not isinstance(data, list):
            raise TypeError(f"Invalid GitHub inventory: {key}")
        result.extend(data)
        if len(data) < 100:
            return result
    raise ValueError("GitHub inventory pagination exceeded limit")


def discover_runners(get, repository, group_name, private):
    base = f"/orgs/{quote(repository.split('/')[0], safe='')}/actions/runner-groups"
    groups = [g for g in collect(get, base, "runner_groups") if g["name"] == group_name]
    if len(groups) != 1:
        raise ValueError(f"Expected one accessible runner group: {group_name}")
    group = groups[0]
    if not private and not group.get("allows_public_repositories", False):
        raise ValueError("Runner group does not allow public repositories")
    if group["visibility"] == "selected":
        repos = collect(get, f"{base}/{group['id']}/repositories", "repositories")
        if not any(r["full_name"].casefold() == repository.casefold() for r in repos):
            raise ValueError("Repository has no access to the runner group")
    if group.get("restricted_to_workflows"):
        raise ValueError(
            "Workflow-restricted runner groups require explicit workflow admission"
        )
    return collect(get, f"{base}/{group['id']}/runners", "runners")


def label_set(runner):
    return {label["name"].casefold() for label in runner["labels"]}


def validate_plan(plan):
    shards = plan["shards"]
    if not 1 <= len(shards) <= 6:
        raise ValueError("Expected one to six DevKit shards")
    if len({s["id"] for s in shards}) != len(shards):
        raise ValueError("Duplicate shard IDs")
    if Counter(t for s in shards for t in s["tasks"]) != Counter(LANES.keys()):
        raise ValueError("Every canonical lane must be assigned exactly once")
    if any(not s["tasks"] or not s["labels"] for s in shards):
        raise ValueError("Empty shard or runner selector")


def make_plan(runners, required, sha, maximum=6, plan_id="local"):
    if not 1 <= maximum <= 6:
        raise ValueError("Fleet cap must be one to six")
    required = {x.casefold() for x in required}
    eligible = [r for r in runners if r["status"] == "online" and required <= label_set(r)]
    if not eligible:
        raise ValueError("No online DevKit matches the exact firmware/capabilities")
    idle = sorted((r for r in eligible if not r["busy"]), key=lambda r: r["name"])
    selected = idle[:maximum] or [min(eligible, key=lambda r: r["name"])]
    shards = [{"id": f"devkit-{i+1}", "candidate": r["name"],
               "labels": sorted(required), "idle_at_planning": not r["busy"],
               "tasks": [], "estimated_seconds": 0} for i,r in enumerate(selected)]
    for task, seconds in sorted(LANES.items(), key=lambda t: (-t[1], t[0])):
        shard = min(shards, key=lambda s: (s["estimated_seconds"], s["id"]))
        shard["tasks"].append(task)
        shard["estimated_seconds"] += seconds
    plan = {"sha": sha, "plan_id": plan_id, "shards": shards}
    validate_plan(plan)
    return plan


def verify_results(plan, directory):
    validate_plan(plan)
    receipts = [json.loads(p.read_text()) for p in directory.rglob("shard-receipt.json")]
    if Counter(r["id"] for r in receipts) != Counter(s["id"] for s in plan["shards"]):
        raise ValueError("Missing or duplicate DevKit receipts")
    for shard in plan["shards"]:
        r = next(r for r in receipts if r["id"] == shard["id"])
        if r["plan_id"] != plan["plan_id"] or r["sha"] != plan["sha"] or r["result"] != "success" or r["tasks"] != shard["tasks"]:
            raise ValueError("Failed, stale, or incomplete DevKit shard")
    partitions = [json.loads(p.read_text()) for p in directory.rglob("e2e-partition.json")]
    if len(partitions) != 2 or {p["index"] for p in partitions} != {0,1}:
        raise ValueError("Missing/duplicate E2E partition accounting")
    if partitions[0]["canonical"] != partitions[1]["canonical"]:
        raise ValueError("Workers discovered different canonical E2E selections")
    assigned = [n for p in partitions for n in p["selected"]]
    if Counter(assigned) != Counter(partitions[0]["canonical"]):
        raise ValueError("E2E partitions omitted or duplicated tests")
    print(f"All {len(receipts)} DevKits passed; all six lanes and {len(assigned)} E2E tests accounted for")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group")
    parser.add_argument("--labels", help="JSON required labels from the firmware resolver")
    parser.add_argument("--output", type=Path, default=Path("devkit-shard-plan.json"))
    parser.add_argument("--verify-results", type=Path)
    args = parser.parse_args()
    if args.verify_results:
        plan = json.loads(args.output.read_text())
        if plan["sha"] != os.environ["GITHUB_SHA"]:
            raise ValueError("Stale shard plan")
        verify_results(plan, args.verify_results)
        return
    token = os.environ["RUNNER_DISCOVERY_TOKEN"]
    def get(path):
        request = Request(os.environ.get("GITHUB_API_URL", "https://api.github.com") + path,
                          headers={"Authorization": f"Bearer {token}",
                                   "Accept": "application/vnd.github+json",
                                   "X-GitHub-Api-Version": "2022-11-28"})
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    runners = discover_runners(get, os.environ["GITHUB_REPOSITORY"], args.group,
                               os.environ.get("REPOSITORY_PRIVATE", "true") == "true")
    plan = make_plan(runners, json.loads(args.labels), os.environ["GITHUB_SHA"],
                     plan_id=f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}")
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    matrix = {"include": [{"shard": s["id"], "plan_id": plan["plan_id"], "labels": s["labels"],
                            "tasks": json.dumps(s["tasks"])} for s in plan["shards"]]}
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"matrix={json.dumps(matrix)}\nplan_id={plan['plan_id']}\n")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write("| Shard | Idle at discovery | Estimated tests | Lanes |\n|---|---|---:|---|\n")
        for s in plan["shards"]:
            f.write(f"| {s['id']} | {s['idle_at_planning']} | {s['estimated_seconds']/60:.1f} min | {', '.join(s['tasks'])} |\n")
        f.write("\nAvailability is a snapshot; GitHub assigns runners and owns queueing. Estimates exclude installation.\n")


if __name__ == "__main__":
    main()
