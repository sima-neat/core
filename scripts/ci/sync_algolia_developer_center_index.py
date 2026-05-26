#!/usr/bin/env python3
"""Generate and sync Developer Center Algolia records for Neat software docs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


SOURCE = "software"
DEFAULT_MAX_RECORD_BYTES = 9_500
DEFAULT_BATCH_SIZE = 500
EXCLUDES = {"_category_.json", "_tmp_test.txt"}
SKIP_PARTS = {"doxygen", "node_modules", "build", "out", ".git"}
EXTENSIONS = {".md", ".mdx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or sync software docs records in the shared Developer Center Algolia index.",
    )
    parser.add_argument("--docs-dir", default="docs", help="Docs root directory.")
    parser.add_argument("--site-base-url", required=True, help="Public Developer Center base URL.")
    parser.add_argument("--records-json", help="Write generated records to this JSON file.")
    parser.add_argument("--input-records-json", help="Read records from this JSON file instead of generating them.")
    parser.add_argument("--summary-json", help="Write generation summary to this JSON file.")
    parser.add_argument("--app-id", help="Algolia app ID.")
    parser.add_argument("--api-key", help="Algolia write API key.")
    parser.add_argument("--index-name", help="Algolia index name.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-record-bytes", type=int, default=DEFAULT_MAX_RECORD_BYTES)
    parser.add_argument("--generate-only", action="store_true", help="Only generate records; do not upload.")
    parser.add_argument("--sync", action="store_true", help="Upload records and remove stale records for this source.")
    return parser.parse_args()


def strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5 :]
    return text


def clean_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def title_from_markdown(path: Path, text: str) -> str:
    for line in text.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line.strip())
        if match:
            return clean_markdown(match.group(1))
    return path.stem.replace("-", " ").replace("_", " ").title()


def section_for_path(path: Path, docs_dir: Path) -> str:
    rel_parts = path.relative_to(docs_dir).parts
    mapping = {
        "getting-started": "Getting Started",
        "how-to": "How-To",
        "reference": "References",
        "tutorials": "Tutorials",
        "contribute": "Contribute",
    }
    for part in rel_parts:
        if part in mapping:
            return mapping[part]
    return "Overview"


def route_for_path(path: Path, docs_dir: Path, site_base_url: str) -> str:
    rel = path.relative_to(docs_dir).as_posix()
    stem = rel.rsplit(".", 1)[0]
    if stem.endswith("/index"):
        stem = stem[: -len("/index")]
    if stem == "index":
        stem = ""

    match = re.match(r"^tutorials/tutorial_v(\d+)_(\d+)_(.+)$", stem)
    if match:
        major, chapter, slug = match.groups()
        stem = f"tutorials/v{major}/{chapter}-{slug.replace('_', '-')}"

    route = f"/software/{stem}".rstrip("/")
    if route == "":
        route = "/software"
    return f"{site_base_url.rstrip('/')}{route}"


def record_size(record: dict) -> int:
    return len(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def fit_record(record: dict, max_record_bytes: int) -> tuple[dict, bool]:
    if record_size(record) <= max_record_bytes:
        return record, False

    content = str(record.get("content", ""))
    lo = 0
    hi = len(content)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = dict(record)
        candidate["content"] = content[:mid]
        candidate["content_truncated"] = True
        if record_size(candidate) <= max_record_bytes:
            best = candidate["content"]
            lo = mid + 1
        else:
            hi = mid - 1

    record["content"] = best
    record["content_truncated"] = True
    if record_size(record) > max_record_bytes:
        record["title"] = str(record.get("title", ""))[:120]
        record["content"] = str(record.get("content", ""))[:400]
        record["content_truncated"] = True
    if record_size(record) > max_record_bytes:
        raise RuntimeError(f"unable to fit record under {max_record_bytes} bytes: {record.get('path')}")
    return record, True


def generate_records(docs_dir: Path, site_base_url: str, max_record_bytes: int) -> tuple[list[dict], dict]:
    docs_dir = docs_dir.resolve()
    records: list[dict] = []
    by_section: dict[str, int] = {}
    trimmed = 0
    max_seen = 0

    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.name in EXCLUDES or path.suffix.lower() not in EXTENSIONS:
            continue
        if any(part in SKIP_PARTS for part in path.relative_to(docs_dir).parts):
            continue

        raw = strip_frontmatter(path.read_text(encoding="utf-8"))
        body = clean_markdown(raw)
        if not body:
            continue

        rel = path.relative_to(docs_dir).as_posix()
        title = title_from_markdown(path, raw)
        section = section_for_path(path, docs_dir)
        url = route_for_path(path, docs_dir, site_base_url)
        route = urllib.parse.urlparse(url).path or "/software"
        record = {
            "objectID": f"{SOURCE}:{hashlib.sha1(rel.encode('utf-8')).hexdigest()}",
            "source": SOURCE,
            "url": url,
            "route": route,
            "path": rel,
            "section": section,
            "category": section,
            "title": title,
            "content": body[:20_000],
            "hierarchy": {"lvl0": section, "lvl1": title},
        }
        record, was_trimmed = fit_record(record, max_record_bytes)
        trimmed += int(was_trimmed)
        max_seen = max(max_seen, record_size(record))
        records.append(record)
        by_section[section] = by_section.get(section, 0) + 1

    summary = {
        "source": SOURCE,
        "count": len(records),
        "by_section": by_section,
        "trimmed_records": trimmed,
        "max_record_bytes": max_seen,
    }
    return records, summary


class AlgoliaClient:
    def __init__(self, app_id: str, api_key: str, index_name: str) -> None:
        self.host = f"https://{app_id}.algolia.net"
        self.index = urllib.parse.quote(index_name, safe="")
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Algolia-Application-Id": app_id,
            "X-Algolia-API-Key": api_key,
        }

    def post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8"),
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code == 404 and path.endswith("/browse"):
                return {"hits": []}
            raise SystemExit(f"Algolia API error: HTTP {error.code} on {path}\n{body}") from error

    def browse_source_object_ids(self, source: str) -> list[str]:
        object_ids: list[str] = []
        payload = {"query": "", "attributesToRetrieve": ["objectID", "source"], "hitsPerPage": 1000}
        while True:
            result = self.post(f"/1/indexes/{self.index}/browse", payload)
            for hit in result.get("hits", []):
                if hit.get("source") == source and hit.get("objectID"):
                    object_ids.append(hit["objectID"])
            cursor = result.get("cursor")
            if not cursor:
                return object_ids
            payload = {"cursor": cursor}

    def batch(self, requests: list[dict]) -> None:
        if not requests:
            return
        result = self.post(f"/1/indexes/{self.index}/batch", {"requests": requests})
        print(f"[algolia-index] taskID={result.get('taskID')} requests={len(requests)}")


def sync_records(args: argparse.Namespace, records: list[dict]) -> None:
    if not args.app_id or not args.api_key or not args.index_name:
        raise SystemExit("--app-id, --api-key, and --index-name are required for --sync")

    client = AlgoliaClient(args.app_id, args.api_key, args.index_name)
    desired_ids = {record["objectID"] for record in records}
    existing_ids = set(client.browse_source_object_ids(SOURCE))
    stale_ids = sorted(existing_ids - desired_ids)

    print(f"[algolia-index] existing {SOURCE} records={len(existing_ids)} stale={len(stale_ids)}")
    for start in range(0, len(stale_ids), args.batch_size):
        chunk = stale_ids[start : start + args.batch_size]
        client.batch([{"action": "deleteObject", "body": {"objectID": object_id}} for object_id in chunk])

    print(f"[algolia-index] uploading {len(records)} {SOURCE} records...")
    for start in range(0, len(records), args.batch_size):
        chunk = records[start : start + args.batch_size]
        client.batch([{"action": "addObject", "body": record} for record in chunk])


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.max_record_bytes < 2_000:
        raise SystemExit("--max-record-bytes must be at least 2000")
    if not args.generate_only and not args.sync:
        args.generate_only = True

    if args.input_records_json:
        records = json.loads(Path(args.input_records_json).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise SystemExit("--input-records-json must contain a JSON list")
        summary = {"source": SOURCE, "count": len(records), "from": args.input_records_json}
        print("[algolia-index] loaded records:")
    else:
        records, summary = generate_records(Path(args.docs_dir), args.site_base_url, args.max_record_bytes)
        print("[algolia-index] generated records:")
    print(json.dumps(summary, indent=2))

    if args.records_json and not args.input_records_json:
        Path(args.records_json).write_text(json.dumps(records, ensure_ascii=True), encoding="utf-8")
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.sync:
        sync_records(args, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
