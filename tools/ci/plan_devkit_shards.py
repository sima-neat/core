#!/usr/bin/env python3
"""Plan DevKit workers by firmware and memory capacity."""
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
         "genai": 129, "genai-python": 100, "performance": 276, "python-tutorials": 310}
MAX_GENERAL_SHARDS = 6
MAX_GENAI_SHARDS = 2

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


def runner_selector(runner, required, blocked):
    """Keep mandatory capabilities and only labels needed to exclude blocked nodes."""
    labels = (label_set(runner) - {"rtsp"}) | required
    if any(labels <= label_set(other) for other in blocked):
        return None
    for label in sorted(labels - required):
        trial = labels - {label}
        if not any(trial <= label_set(other) for other in blocked):
            labels = trial
    return sorted(labels)


def expected_lanes(weekly=False):
    return {**LANES, **({"streams": 120} if weekly else {})}


def task_pool(task):
    if task in ("genai", "genai-python"):
        return "llm-16gb"
    return "default"


def validate_plan(plan):
    shards = plan["shards"]
    expected = expected_lanes(plan.get("weekly", False))
    maximum_shards = MAX_GENERAL_SHARDS + MAX_GENAI_SHARDS
    if not 1 <= len(shards) <= maximum_shards:
        raise ValueError(f"Expected 1 to {maximum_shards} DevKit shards")
    if len({s["id"] for s in shards}) != len(shards):
        raise ValueError("Duplicate shard IDs")
    if Counter(t for s in shards for t in s["tasks"]) != Counter(expected.keys()):
        raise ValueError("Every canonical lane must be assigned exactly once")
    for pool, maximum in (("default", MAX_GENERAL_SHARDS), ("llm-16gb", MAX_GENAI_SHARDS)):
        if sum(s["pool"] == pool for s in shards) > maximum:
            raise ValueError("Capability pool exceeds its fleet cap")
    for shard in shards:
        if not shard["tasks"] or not shard["labels"]:
            raise ValueError("Empty shard or runner selector")
        pools = {task_pool(t) for t in shard["tasks"]}
        if pools != {shard["pool"]}:
            raise ValueError("Lane assigned to the wrong capability pool")
        labels = {x.casefold() for x in shard["labels"]}
        if shard["pool"] == "llm-16gb" and "modalix-som-16gig" not in labels:
            raise ValueError("LLM tests require modalix-som-16gig")
        if "rtsp" in labels:
            raise ValueError("DevKit selectors must not require the legacy RTSP label")


def make_plan(runners, required, sha, maximum=MAX_GENERAL_SHARDS, plan_id="local", *, weekly=False,
              stream_profile="http", excluded_names=()):
    if not 1 <= maximum <= MAX_GENERAL_SHARDS:
        raise ValueError(f"Default fleet cap must be 1 to {MAX_GENERAL_SHARDS}")
    if weekly and stream_profile not in ("none", "http", "all"):
        raise ValueError("Unknown stream capability profile")
    required = {x.casefold() for x in required}
    if "rtsp" in required:
        raise ValueError("Base runner selector must not require RTSP")
    expected = expected_lanes(weekly)
    excluded = [r for r in runners if r["name"].casefold() in {x.casefold() for x in excluded_names}]
    shards, reserved = [], set()
    # Admit scarce memory-specific workers first. Count each candidate once.
    for pool, limit in (("llm-16gb", MAX_GENAI_SHARDS), ("default", maximum)):
        tasks = {t: w for t, w in expected.items() if task_pool(t) == pool}
        if not tasks:
            continue
        candidates, selectors = [], {}
        pool_required = required | ({"modalix-som-16gig"} if pool == "llm-16gb" else set())
        for r in runners:
            if (r["status"] != "online" or not required <= label_set(r)
                    or (pool == "llm-16gb" and "modalix-som-16gig" not in label_set(r))):
                continue
            labels = runner_selector(r, pool_required, excluded)
            if labels is None:
                continue
            selectors[r["name"]] = labels
            candidates.append(r)
        if not candidates:
            raise ValueError(f"No safe online {pool} DevKit matches the firmware/capabilities")
        idle = sorted((r for r in candidates if not r["busy"] and r["name"] not in reserved),
                      key=lambda r: r["name"])
        selected = idle[:min(limit, len(tasks))]
        if not selected:
            # Queue one complete pool when idle capacity is unavailable.
            # A shared selector still permits GitHub to assign any eligible node.
            selected = [min(candidates, key=lambda r: (r["name"] in reserved, r["name"]))]
        bins = []
        for r in selected:
            reserved.add(r["name"])
            bins.append({"id": f"{pool}-{len(bins)+1}", "pool": pool,
                         "candidate": r["name"], "labels": selectors[r["name"]],
                         "idle_at_planning": not r["busy"], "tasks": [], "estimated_seconds": 0})
        for task, seconds in sorted(tasks.items(), key=lambda t: (-t[1], t[0])):
            shard = min(bins, key=lambda s: (s["estimated_seconds"], s["id"]))
            shard["tasks"].append(task)
            shard["estimated_seconds"] += seconds
        shards.extend(bins)
    plan = {"sha": sha, "plan_id": plan_id, "weekly": weekly,
            "stream_profile": stream_profile, "shards": shards}
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
    print(f"All {len(receipts)} shards passed; all {len(expected_lanes(plan.get('weekly', False)))} lanes and {len(assigned)} E2E tests accounted for")


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
    excluded = json.loads(os.environ.get("CORE_UNREADY_RUNNERS") or "[]")
    if not isinstance(excluded, list) or any(not isinstance(n, str) for n in excluded):
        raise ValueError("CORE_UNREADY_RUNNERS must be a JSON array of names")
    plan = make_plan(runners, json.loads(args.labels), os.environ["GITHUB_SHA"],
                     plan_id=f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}",
                     weekly=os.environ.get("RUN_LONG_TESTS") == "true",
                     stream_profile=os.environ.get("STREAM_PROFILE", "http"),
                     excluded_names=excluded)
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
