#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DOCS_DIR="${DOCS_DIR:-docs}"
SITE_BASE_URL="${SITE_BASE_URL:-https://neat.modalix.info}"
BATCH_SIZE="${BATCH_SIZE:-500}"
MAX_RECORD_BYTES="${MAX_RECORD_BYTES:-9500}"
DRY_RUN="0"
CLEAR_INDEX="0"
VERBOSE="0"

usage() {
  cat <<'USAGE'
Usage: bash scripts/ci/sync_algolia_docs_index.sh [options]

Index docs markdown files into Algolia without crawler.

Options:
  --docs-dir <path>      Docs root directory (default: docs)
  --site-base-url <url>  Public docs base URL (default: https://neat.modalix.info)
  --batch-size <n>       Upload batch size (default: 500)
  --max-record-bytes <n> Max per-record payload size in bytes (default: 9500)
  --clear                Clear target index before upload
  --dry-run              Generate records only; do not upload
  --verbose              Print extra diagnostics
  -h, --help             Show this help

Env vars:
  ALGOLIA_APP_ID or DOCS_ALGOLIA_APP_ID
  ALGOLIA_ADMIN_API_KEY
  ALGOLIA_INDEX_NAME or DOCS_ALGOLIA_INDEX_NAME

Examples:
  bash scripts/ci/sync_algolia_docs_index.sh --dry-run --verbose

  ALGOLIA_APP_ID=xxx ALGOLIA_ADMIN_API_KEY=yyy ALGOLIA_INDEX_NAME=zzz \
  bash scripts/ci/sync_algolia_docs_index.sh --clear
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --docs-dir)
      DOCS_DIR="${2:-}"
      shift 2
      ;;
    --site-base-url)
      SITE_BASE_URL="${2:-}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --max-record-bytes)
      MAX_RECORD_BYTES="${2:-}"
      shift 2
      ;;
    --clear)
      CLEAR_INDEX="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --verbose)
      VERBOSE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "${DOCS_DIR}" ]]; then
  echo "Docs directory not found: ${DOCS_DIR}" >&2
  exit 1
fi

if ! [[ "${BATCH_SIZE}" =~ ^[0-9]+$ ]] || [[ "${BATCH_SIZE}" -lt 1 ]]; then
  echo "Invalid --batch-size value: ${BATCH_SIZE}" >&2
  exit 1
fi
if ! [[ "${MAX_RECORD_BYTES}" =~ ^[0-9]+$ ]] || [[ "${MAX_RECORD_BYTES}" -lt 2000 ]]; then
  echo "Invalid --max-record-bytes value: ${MAX_RECORD_BYTES}" >&2
  exit 1
fi

# Docusaurus docs in this repo use routeBasePath="/", so normalize accidental
# /docs base URLs to canonical site root.
SITE_BASE_URL="${SITE_BASE_URL%/}"
if [[ "${SITE_BASE_URL}" == */docs ]]; then
  SITE_BASE_URL="${SITE_BASE_URL%/docs}"
fi

TMP_DIR="$(mktemp -d /tmp/algolia-docs-index-XXXXXX)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

RECORDS_JSON="${TMP_DIR}/records.json"
SUMMARY_JSON="${TMP_DIR}/summary.json"

python3 - "$DOCS_DIR" "$SITE_BASE_URL" "$RECORDS_JSON" "$SUMMARY_JSON" "$VERBOSE" "$MAX_RECORD_BYTES" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

docs_dir = Path(sys.argv[1]).resolve()
site_base_url = sys.argv[2].rstrip("/")
records_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
verbose = sys.argv[5] == "1"
max_record_bytes = int(sys.argv[6])

EXCLUDES = {"_category_.json", "_tmp_test.txt"}
SKIP_PARTS = {"doxygen", "node_modules", "build", "out", ".git"}
EXTS = {".md", ".mdx"}

def clean_md(text: str) -> str:
    # Strip code fences.
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Strip inline code/backticks.
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Strip markdown links to label.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Strip html tags.
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text

def strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5 :]
    return text

def doc_url(path: Path) -> str:
    rel = path.relative_to(docs_dir).as_posix()
    stem = rel
    if stem.endswith(".mdx"):
        stem = stem[:-4]
    elif stem.endswith(".md"):
        stem = stem[:-3]
    if stem.endswith("/index"):
        stem = stem[: -len("/index")]
    if stem == "index":
        stem = ""

    # Canonical tutorials route mapping:
    # docs/tutorials/tutorial_v2_007_session_patterns.mdx
    # -> /tutorials/v2/007-session-patterns
    m = re.match(r"^tutorials/tutorial_v(\d+)_(\d+)_(.+)$", stem)
    if m:
        major, chapter, slug = m.groups()
        stem = f"tutorials/v{major}/{chapter}-{slug.replace('_', '-')}"

    return f"{site_base_url}/{stem}".rstrip("/") or site_base_url

def section_of(path: Path) -> str:
    rel_parts = path.relative_to(docs_dir).parts
    if not rel_parts:
        return "Other"
    mapping = {
        "getting-started": "Getting Started",
        "how-to": "How-To",
        "reference": "References",
        "tutorials": "Tutorials",
        "contribute": "Contribute",
    }
    for part in rel_parts:
        sec = mapping.get(part)
        if sec:
            return sec
    return "Other"

def read_title_and_body(path: Path):
    raw = path.read_text(encoding="utf-8")
    raw = strip_frontmatter(raw)
    title = None
    lines = raw.splitlines()
    for line in lines:
        m = re.match(r"^#\s+(.+?)\s*$", line.strip())
        if m:
            title = m.group(1).strip()
            break
    if title is None:
        title = path.stem
    body = clean_md(raw)
    return title, body

records = []
by_section = {}
trimmed_records = 0
max_seen_bytes = 0

def record_size_bytes(rec) -> int:
    return len(json.dumps(rec, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

def fit_record_to_size(rec):
    global trimmed_records, max_seen_bytes
    sz = record_size_bytes(rec)
    if sz <= max_record_bytes:
        max_seen_bytes = max(max_seen_bytes, sz)
        return rec

    # Trim content iteratively until record fits within size budget.
    content = rec.get("content", "")
    if not isinstance(content, str):
        content = str(content)

    lo, hi = 0, len(content)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = dict(rec)
        candidate["content"] = content[:mid]
        candidate["content_truncated"] = True
        csz = record_size_bytes(candidate)
        if csz <= max_record_bytes:
            best = candidate["content"]
            lo = mid + 1
        else:
            hi = mid - 1

    rec["content"] = best
    rec["content_truncated"] = True
    final_sz = record_size_bytes(rec)
    if final_sz > max_record_bytes:
        # Extreme fallback: shrink title/content aggressively.
        rec["title"] = str(rec.get("title", ""))[:120]
        rec["content"] = str(rec.get("content", ""))[:400]
        rec["content_truncated"] = True
        final_sz = record_size_bytes(rec)

    if final_sz > max_record_bytes:
        raise RuntimeError(
            f"unable to fit record under {max_record_bytes} bytes for path={rec.get('path')}"
        )
    trimmed_records += 1
    max_seen_bytes = max(max_seen_bytes, final_sz)
    return rec

for path in sorted(docs_dir.rglob("*")):
    if not path.is_file():
        continue
    if path.name in EXCLUDES:
        continue
    if path.suffix.lower() not in EXTS:
        continue
    if any(p in SKIP_PARTS for p in path.relative_to(docs_dir).parts):
        continue

    title, body = read_title_and_body(path)
    if not body:
        continue

    rel = path.relative_to(docs_dir).as_posix()
    sec = section_of(path)
    object_id = hashlib.sha1(rel.encode("utf-8")).hexdigest()
    rec = {
        "objectID": object_id,
        "url": doc_url(path),
        "path": rel,
        "section": sec,
        "title": title,
        "content": body[:20000],
        "hierarchy": {"lvl0": sec, "lvl1": title},
    }
    records.append(fit_record_to_size(rec))
    by_section[sec] = by_section.get(sec, 0) + 1

records_path.write_text(json.dumps(records, ensure_ascii=True), encoding="utf-8")
summary_path.write_text(
    json.dumps(
        {
            "count": len(records),
            "by_section": by_section,
            "trimmed_records": trimmed_records,
            "max_record_bytes": max_seen_bytes,
        },
        indent=2,
    ),
    encoding="utf-8",
)

if verbose:
    print(
        json.dumps(
            {
                "count": len(records),
                "by_section": by_section,
                "trimmed_records": trimmed_records,
                "max_record_bytes": max_seen_bytes,
            },
            indent=2,
        )
    )
PY

echo "[algolia-index] generated records:"
cat "${SUMMARY_JSON}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[algolia-index] dry-run only; skipping upload."
  exit 0
fi

ALGOLIA_APP_ID="${ALGOLIA_APP_ID:-${DOCS_ALGOLIA_APP_ID:-}}"
ALGOLIA_ADMIN_API_KEY="${ALGOLIA_ADMIN_API_KEY:-}"
ALGOLIA_INDEX_NAME="${ALGOLIA_INDEX_NAME:-${DOCS_ALGOLIA_INDEX_NAME:-}}"

if [[ -z "${ALGOLIA_APP_ID}" || -z "${ALGOLIA_ADMIN_API_KEY}" || -z "${ALGOLIA_INDEX_NAME}" ]]; then
  echo "Missing required Algolia credentials." >&2
  echo "Set ALGOLIA_APP_ID (or DOCS_ALGOLIA_APP_ID), ALGOLIA_ADMIN_API_KEY, ALGOLIA_INDEX_NAME (or DOCS_ALGOLIA_INDEX_NAME)." >&2
  exit 1
fi

if [[ "${ALGOLIA_INDEX_NAME}" == "${ALGOLIA_APP_ID}" ]]; then
  echo "ALGOLIA_INDEX_NAME appears to be set to the App ID (${ALGOLIA_APP_ID})." >&2
  echo "Set ALGOLIA_INDEX_NAME to your index (for example: neat_modalix_info_k73g7gy9dh_articles)." >&2
  exit 1
fi

python3 - "$ALGOLIA_APP_ID" "$ALGOLIA_ADMIN_API_KEY" "$ALGOLIA_INDEX_NAME" "$RECORDS_JSON" "$BATCH_SIZE" "$CLEAR_INDEX" <<'PY'
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

app_id = sys.argv[1]
admin_key = sys.argv[2]
index_name = sys.argv[3]
records_path = sys.argv[4]
batch_size = int(sys.argv[5])
clear_index = sys.argv[6] == "1"

host = f"https://{app_id}.algolia.net"
idx = urllib.parse.quote(index_name, safe="")
headers = {
    "X-Algolia-Application-Id": app_id,
    "X-Algolia-API-Key": admin_key,
    "Content-Type": "application/json",
}

def api_post(path: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(host + path, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = "<unable to read response body>"
        msg = (
            f"Algolia API error: HTTP {e.code} {e.reason} on {path}\n"
            f"Response body: {body}"
        )
        raise SystemExit(msg)

records = json.loads(open(records_path, "r", encoding="utf-8").read())
if not isinstance(records, list):
    raise SystemExit("records payload is not a list")

if clear_index:
    print("[algolia-index] clearing index...")
    resp = api_post(f"/1/indexes/{idx}/clear", {})
    print(f"[algolia-index] clear taskID={resp.get('taskID')}")

total = len(records)
print(f"[algolia-index] uploading {total} records to index={index_name} in batches of {batch_size}...")

offset = 0
batch_no = 0
while offset < total:
    batch_no += 1
    chunk = records[offset : offset + batch_size]
    requests = [{"action": "addObject", "body": rec} for rec in chunk]
    resp = api_post(f"/1/indexes/{idx}/batch", {"requests": requests})
    print(
        f"[algolia-index] batch={batch_no} uploaded={len(chunk)} "
        f"taskID={resp.get('taskID')}"
    )
    offset += len(chunk)

print("[algolia-index] done.")
PY
