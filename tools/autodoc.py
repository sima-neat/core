#!/usr/bin/env python3
"""Pull docs from sibling repos and splice them into core's docs tree.

Reads a JSON manifest of source repos. For each entry, clones (or updates)
into a staging dir under ${BUILD_DIR}/autodoc/<key>/, then copies the
configured docs subpath into <out_root>/<mount>/ with light transforms:

- Inject YAML frontmatter (title, sidebar_position) if missing.
- Rewrite sibling .md links to Docusaurus-style routes.
- Rewrite configured imported-doc link targets to their mounted core routes.
- Drop a generated _category_.json at the section root.

Per-source clone or copy failures are logged and skipped; the script always
exits 0 unless the manifest itself is malformed.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger("autodoc")

FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
# griffe section labels at column 0 — promoted to H2 headings so Docusaurus'
# on-this-page TOC picks them up and so they can be styled per-section.
GRIFFE_TOP_LABEL_RE = re.compile(
    r"^(Imports|Classes|Functions|Constants):\s*$", re.MULTILINE,
)
# Label-alone nested labels (e.g. `    Parameters:` with the value list below).
# These need a blank line before AND after, otherwise markdown folds them into
# the surrounding bullet's paragraph.
GRIFFE_LABEL_ALONE_RE = re.compile(
    r"^(\s+)(Parameters):\s*$", re.MULTILINE,
)
# Inline-value nested labels (e.g. `    Returns: list[np.ndarray]`). Need a
# blank line before so the label doesn't get absorbed into the preceding
# bullet's text run; the value stays on the same line.
GRIFFE_LABEL_WITH_VALUE_RE = re.compile(
    r"^(\s+)(Returns|Decorators):(?=\s)", re.MULTILINE,
)
# Inline `Decorators:` that lives mid-line inside a method bullet's signature.
GRIFFE_INLINE_DECORATORS_RE = re.compile(r"(?<!\*\*)Decorators:")
# Match [label](target.md) and [label](target.md#anchor); skip absolute / external.
MD_LINK_RE = re.compile(r"(\]\()(?!https?:|/|#)([^)\s]+?)\.md(#[^)\s]+)?(\))")
MARKDOWN_TARGET_RE = re.compile(r"(?P<prefix>\]\()(?P<target>[^)\s]+)(?P<suffix>\))")
HTML_HREF_TARGET_RE = re.compile(r"(?P<prefix>\bhref=[\"'])(?P<target>[^\"']+)(?P<suffix>[\"'])")
# Match a per-module entry in a source flat index:
#   - [`afe.apis.foo`](afe-apis-foo.md) (3 functions) - summary text
SOURCE_INDEX_ENTRY_RE = re.compile(
    r"^-\s+\[`(?P<name>[^`]+)`\]\((?P<file>[^)]+)\.md\)"
    r"(?:\s+\((?P<counts>[^)]+)\))?"
    r"(?:\s+-\s+(?P<summary>.+?))?\s*$"
)


def run_git(args: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    """Run git with stdout/stderr captured; raise CalledProcessError on failure."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=run_env,
    )


def github_token() -> str:
    """Best-effort token for GitHub HTTPS clones in CI."""
    for env_var in ("AUTODOC_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    return ""


def github_auth_env() -> Dict[str, str]:
    """Return temporary git config env for GitHub token auth, without URL tokens."""
    token = github_token()
    if not token:
        return {}

    encoded = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {encoded}",
    }


def github_https_repo(repo: str) -> str:
    """Rewrite GitHub SSH remotes to HTTPS when token auth is available."""
    if not github_token():
        return repo
    match = re.match(r"^git@github\.com:(?P<path>[^\s]+?)(?:\.git)?$", repo)
    if match:
        return f"https://github.com/{match.group('path')}.git"
    return repo


def is_git_checkout(path: Path) -> bool:
    """Return True when `path` is an existing usable git working tree."""
    if not path.is_dir():
        return False
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(path),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def clone_source(repo: str, branches: List[str], githash: str, staging: Path) -> str:
    """Clone `repo` into a fresh `staging` directory.

    Tries each candidate branch in order and returns the one that clones
    successfully. The clone itself decides which branch exists — there is no
    separate remote probe — so a missing snap branch cleanly falls through to
    the next candidate, while a genuine auth/network failure surfaces as the
    raised error instead of being silently misread as "branch absent".
    """
    clone_repo = github_https_repo(repo)
    auth_env = github_auth_env()
    staging.parent.mkdir(parents=True, exist_ok=True)
    if githash:
        run_git(["clone", clone_repo, str(staging)], env=auth_env)
        run_git(["checkout", githash], cwd=staging, env=auth_env)
        return githash
    last_exc: Optional[subprocess.CalledProcessError] = None
    for branch in branches:
        try:
            run_git(["clone", "--branch", branch, "--depth", "1", clone_repo, str(staging)], env=auth_env)
            return branch
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            if staging.exists():
                shutil.rmtree(staging)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("no branch candidates to clone")


def acquire_source(repo: str, branches: List[str], githash: str, staging: Path) -> str:
    """Clone or update `repo` into `staging`; return the branch (or githash) used.

    `branches` is the ordered candidate list; the first that resolves against the
    remote wins. Branch selection therefore happens as part of acquiring the repo,
    not via a pre-flight probe.
    """
    if not is_git_checkout(staging):
        if staging.exists():
            LOG.info("removing stale autodoc staging directory: %s", staging)
            shutil.rmtree(staging)
        return clone_source(repo, branches, githash, staging)

    try:
        if githash:
            run_git(["fetch", "origin"], cwd=staging, env=github_auth_env())
            run_git(["checkout", githash], cwd=staging, env=github_auth_env())
            run_git(["reset", "--hard", githash], cwd=staging, env=github_auth_env())
            return githash
        last_exc: Optional[subprocess.CalledProcessError] = None
        for branch in branches:
            try:
                run_git(["fetch", "--depth", "1", "origin", branch], cwd=staging, env=github_auth_env())
                run_git(["reset", "--hard", "FETCH_HEAD"], cwd=staging, env=github_auth_env())
                return branch
            except subprocess.CalledProcessError as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("no branch candidates to fetch")
    except subprocess.CalledProcessError:
        LOG.info("refresh failed; recloning autodoc staging directory: %s", staging)
        shutil.rmtree(staging)
        return clone_source(repo, branches, githash, staging)


def current_core_branch(repo_root: Path) -> str:
    """Best-effort current branch of the core repo.

    Mirrors build.sh's `current_core_branch`: prefer CI-provided refs, then fall
    back to the local git checkout. Returns "" when it can't be determined.
    """
    for env_var in ("GITHUB_HEAD_REF", "GITHUB_REF_NAME"):
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        ).stdout.strip()
        return "" if out == "HEAD" else out
    except Exception:
        return ""


def resolve_branch_candidates(source: Dict, repo_root: Path) -> Tuple[List[str], Optional[str]]:
    """Ordered branch candidates to try at clone time.

    Returns `(candidates, snap_target)`. `snap_target` is `None` for non-snap
    sources, the current core branch for snap sources (or `""` if it can't be
    determined). The clone tries each candidate in order and the first that
    resolves against the remote wins — there is no separate `ls-remote` probe,
    so a missing snap branch falls through to the next candidate while a real
    auth/network failure is not silently misread as a branch fallback.

    With `"branch_policy": "snap"`, candidates are: current core branch, then
    `fallback_branch` (default "main"), then the configured `branch`.
    """
    configured = source.get("branch", "main")
    if str(source.get("branch_policy", "")).strip().lower() != "snap":
        return [configured], None

    fallback = source.get("fallback_branch", "main")
    core_branch = current_core_branch(repo_root)
    candidates: List[str] = []
    for candidate in (core_branch, fallback, configured):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates, (core_branch or "")


def has_frontmatter(text: str) -> bool:
    return bool(FRONTMATTER_RE.match(text))


def derive_title(text: str, stem: str) -> str:
    match = H1_RE.search(text)
    if match:
        # Strip surrounding backticks left over from source H1s like `# `afe.apis.foo``.
        return match.group(1).strip().strip("`").strip()
    # Fallback: humanize the file stem.
    return stem.replace("-", " ").replace("_", " ").strip().title() or stem


def derive_sidebar_label(title: str) -> str:
    """Short sidebar entry. For dotted module paths, use the trailing segment."""
    if "." in title:
        return title.rsplit(".", 1)[-1]
    return title


def inject_frontmatter(
    text: str,
    stem: str,
    position: Optional[int],
    sidebar_label: Optional[str] = None,
) -> str:
    if has_frontmatter(text):
        return text
    title = derive_title(text, stem)
    lines = ["---", f"title: {json.dumps(title)}"]
    if sidebar_label is None:
        sidebar_label = derive_sidebar_label(title)
    if sidebar_label and sidebar_label != title:
        lines.append(f"sidebar_label: {json.dumps(sidebar_label)}")
    if position is not None:
        lines.append(f"sidebar_position: {position}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + "\n" + text


def group_slug(label: str) -> str:
    """Folder-safe slug from a group label."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-").lower()
    return slug or "group"


def build_group_index(landing: Optional[Dict]) -> Dict[str, Tuple[str, str, int, int]]:
    """Map module-stem -> (folder_slug, group_label, group_position, position_within_group).

    Returns an empty dict if landing is None or has no groups.
    """
    out: Dict[str, Tuple[str, str, int, int]] = {}
    if not landing:
        return out
    for group_idx, group in enumerate(landing.get("groups", []), start=1):
        slug = group.get("dir") or group_slug(group["label"])
        for module_idx, module_stem in enumerate(group.get("modules", []), start=1):
            out[module_stem] = (slug, group["label"], group_idx, module_idx)
    return out


def restructure_api_page(text: str) -> str:
    """Promote griffe-style labels into headings / bold callouts.

    Source pages emit plain `Imports:`, `Classes:`, etc. as line-level labels,
    which Docusaurus would otherwise render as inert text. Promote them to H2
    so the on-this-page TOC surfaces them and so they can be color-scoped via
    CSS (see website/src/css/custom.css). Nested `Parameters:`, `Returns:`,
    `Decorators:` labels become bold inline callouts so they're easy to
    spot inside long method bullets.
    """
    text = GRIFFE_TOP_LABEL_RE.sub(r"## \1", text)
    # Blank line before AND after the label-only `Parameters:` so it renders
    # as its own paragraph and the bullets below are recognized as a list.
    text = GRIFFE_LABEL_ALONE_RE.sub(r"\n\1**\2:**\n", text)
    # Blank line before `Returns: ...` / `Decorators: ...` so they're not
    # absorbed into the preceding bullet's paragraph; the value stays inline.
    text = GRIFFE_LABEL_WITH_VALUE_RE.sub(r"\n\1**\2:**", text)
    text = GRIFFE_INLINE_DECORATORS_RE.sub("**Decorators:**", text)
    return text


def rewrite_md_links(
    text: str,
    current_folder: str,
    stem_to_folder: Dict[str, str],
) -> str:
    """Rewrite sibling `.md` links so cross-folder targets stay resolvable.

    `current_folder` is the folder (relative to the section root) holding the
    file being transformed. `stem_to_folder` maps every staged module stem to
    the folder it now lives in (empty string = section root).

    The `.md` extension is preserved so Docusaurus' MDX link resolver maps
    each link to its routed absolute URL at build time. Stripping `.md` would
    leave the link as a literal relative URL, which the browser resolves
    against Docusaurus' trailing-slash routes one level too deep.
    """
    def repl(match: re.Match[str]) -> str:
        prefix = match.group(1)
        target = match.group(2)
        anchor = match.group(3) or ""
        suffix = match.group(4)
        if "/" in target:
            # Sub-path link; preserve relative structure and the .md.
            return f"{prefix}{target}.md{anchor}{suffix}"
        target_folder = stem_to_folder.get(target, current_folder)
        if target_folder == current_folder:
            return f"{prefix}{target}.md{anchor}{suffix}"
        if current_folder == "":
            return f"{prefix}{target_folder}/{target}.md{anchor}{suffix}"
        if target_folder == "":
            return f"{prefix}../{target}.md{anchor}{suffix}"
        return f"{prefix}../{target_folder}/{target}.md{anchor}{suffix}"
    return MD_LINK_RE.sub(repl, text)


def normalize_link_rewrites(raw_rewrites: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """Validate and normalize source-specific imported-doc link rewrites."""
    rewrites: List[Tuple[str, str]] = []
    for entry in raw_rewrites:
        if not isinstance(entry, dict):
            raise ValueError("link_rewrites entries must be objects with 'from' and 'to'")
        source = str(entry.get("from", "")).strip()
        target = str(entry.get("to", "")).strip()
        if not source or not target:
            raise ValueError("link_rewrites entries require non-empty 'from' and 'to'")
        rewrites.append((source, target))
    return rewrites


def rewrite_configured_link_targets(text: str, link_rewrites: List[Tuple[str, str]]) -> str:
    """Rewrite exact Markdown/HTML link targets using source-specific mapping.

    Autodoc mounts docs from sibling repos under core's information architecture.
    Some imported pages contain absolute or directory-style links that are valid
    in the source repo but not after mounting. Keep those adjustments declarative
    in the manifest and only rewrite link targets, not arbitrary prose/code text.
    """
    if not link_rewrites:
        return text

    def rewrite_target(target: str) -> str:
        for source, replacement in link_rewrites:
            if target == source:
                return replacement
            if target.startswith(f"{source}#"):
                return f"{replacement}{target[len(source):]}"
            if target.startswith(f"{source}?"):
                return f"{replacement}{target[len(source):]}"
        return target

    def repl(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{rewrite_target(match.group('target'))}{match.group('suffix')}"

    text = MARKDOWN_TARGET_RE.sub(repl, text)
    return HTML_HREF_TARGET_RE.sub(repl, text)


def position_for(stem: str, files_order: List[str]) -> Optional[int]:
    if stem in files_order:
        # 1-based: explicit ordering wins.
        return files_order.index(stem) + 1
    if files_order:
        # Files not listed sort after listed ones, alphabetically among themselves.
        return len(files_order) + 1
    return None


def copy_section(
    src_root: Path,
    dst_root: Path,
    files_order: List[str],
    group_index: Dict[str, Tuple[str, str, int, int]],
    exclude_files: Optional[List[str]] = None,
    restructure_api: bool = False,
    link_rewrites: Optional[List[Tuple[str, str]]] = None,
) -> int:
    """Copy src_root into dst_root, transforming markdown.

    When `group_index` contains an entry for a top-level module stem, that
    file is placed under `dst_root/<folder_slug>/` instead of `dst_root/`.
    Top-level markdown stems listed in `exclude_files` are skipped entirely
    (and link rewrites to them are left as-is).
    Returns the number of files staged.
    """
    excluded = set(exclude_files or [])
    link_rewrites = link_rewrites or []
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    # First pass: discover the final folder for every top-level markdown stem
    # so link rewrites in the second pass can compute correct relative paths.
    stem_to_folder: Dict[str, str] = {}
    plan: List[Tuple[Path, Path, str]] = []  # (src, dst, current_folder_rel)
    for src_path in sorted(src_root.rglob("*")):
        if src_path.is_dir():
            continue
        rel = src_path.relative_to(src_root)
        is_top_level_md = src_path.parent == src_root and src_path.suffix.lower() in {".md", ".mdx"}
        if is_top_level_md and src_path.stem in excluded:
            continue
        if is_top_level_md and src_path.stem in group_index:
            folder = group_index[src_path.stem][0]
            dst_path = dst_root / folder / rel.name
            stem_to_folder[src_path.stem] = folder
            current_folder = folder
        else:
            dst_path = dst_root / rel
            current_folder = str(rel.parent) if rel.parent != Path(".") else ""
            if is_top_level_md:
                stem_to_folder[src_path.stem] = ""
        plan.append((src_path, dst_path, current_folder))

    # Second pass: actually copy + transform.
    count = 0
    for src_path, dst_path, current_folder in plan:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.suffix.lower() in {".md", ".mdx"}:
            text = src_path.read_text(encoding="utf-8")
            stem = src_path.stem
            if stem in group_index:
                _, _, _, position = group_index[stem]
            elif src_path.parent == src_root:
                position = position_for(stem, files_order)
            else:
                position = None
            text = inject_frontmatter(text, stem, position)
            if restructure_api:
                text = restructure_api_page(text)
            text = rewrite_md_links(text, current_folder, stem_to_folder)
            text = rewrite_configured_link_targets(text, link_rewrites)
            dst_path.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(src_path, dst_path)
        count += 1
    return count


def write_group_categories(dst_root: Path, landing: Optional[Dict]) -> None:
    """Drop `_category_.json` into each group folder.

    Groups are explicitly non-linkable (`link: null`) — without this Docusaurus
    auto-links the category to its first child, so clicking the category
    navigates away instead of just toggling the section open.
    """
    if not landing:
        return
    for group_idx, group in enumerate(landing.get("groups", []), start=1):
        slug = group.get("dir") or group_slug(group["label"])
        folder = dst_root / slug
        if not folder.is_dir():
            continue
        payload = {
            "label": group["label"],
            # +1 so the landing page (sidebar_position 1) always sits above groups.
            "position": group_idx + 1,
            "link": None,
        }
        (folder / "_category_.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def write_category_json(dst_root: Path, label: str, position: int) -> None:
    payload = {"label": label, "position": position}
    (dst_root / "_category_.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def set_frontmatter_field(text: str, field: str, value: str) -> str:
    """Insert or replace a scalar field in existing YAML frontmatter (no-op if none)."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return text
    fm = match.group(0)
    body = text[match.end():]
    line = f"{field}: {value}"
    if re.search(rf"^{re.escape(field)}:", fm, re.MULTILINE):
        fm = re.sub(rf"^{re.escape(field)}:.*$", line, fm, flags=re.MULTILINE)
    else:
        idx = fm.rfind("---")
        fm = fm[:idx] + line + "\n" + fm[idx:]
    return fm + body


def regroup_command_pages(dst_section: Path, config: Dict) -> None:
    """Reorganize a flat CLI command reference into a subsection per top-level command.

    Command pages follow the convention `<prefix><top>[-<sub>...].md` (e.g.
    `sima-cli-modelzoo-get.md`). Each top-level command `<top>` becomes a folder
    `<dst_section>/<top>/` whose category index is the parent command page
    (`<prefix><top>.md` → `<top>/index.md`) and whose other members are that
    command's subcommand pages. The bare program page (`<root_stem>.md`) is moved
    to the section root. All sibling/cross-page `.md` links are then recomputed
    against the new layout. The section's landing `index.md` is left in place.
    """
    prefix = config.get("prefix", "")
    root_stem = config.get("root_stem", "")
    position_base = int(config.get("position_base", 3))
    # The bare program page must not be named after the section folder: Docusaurus
    # treats a `<folder>.md` as the category index and it would collide with the
    # landing `index.md` (duplicate route). Allow an override, else auto-disambiguate.
    root_page_stem = config.get("root_page_stem") or root_stem
    if root_page_stem == dst_section.name:
        root_page_stem = f"{root_page_stem}-cli"

    # 1. Discover command pages anywhere under the section (skip the landing index).
    command_files: List[Tuple[Path, str]] = []
    for path in dst_section.rglob("*.md"):
        if path.name == "index.md" and path.parent == dst_section:
            continue
        stem = path.stem
        if stem == root_stem or (prefix and stem.startswith(prefix)):
            command_files.append((path, stem))

    # 2. Compute each page's new location (relative to dst_section) + group membership.
    stem_to_target: Dict[str, str] = {}
    groups: Dict[str, List[str]] = {}
    for _, stem in command_files:
        if stem == root_stem:
            stem_to_target[stem] = f"{root_page_stem}.md"
            continue
        rest = stem[len(prefix):]
        top = rest.split("-", 1)[0]
        groups.setdefault(top, [])
        if rest == top:
            stem_to_target[stem] = f"{top}/index.md"  # parent command → category index
        else:
            stem_to_target[stem] = f"{top}/{stem}.md"
            groups[top].append(stem)

    # 3. Move files into their new homes.
    src_by_stem = {stem: path for path, stem in command_files}
    for stem, target in stem_to_target.items():
        src = src_by_stem[stem]
        dst = dst_section / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.move(str(src), str(dst))

    # 4. Keep the bare program page near the top of the section.
    root_page = dst_section / f"{root_page_stem}.md"
    if root_page.is_file():
        text = root_page.read_text(encoding="utf-8")
        root_page.write_text(set_frontmatter_field(text, "sidebar_position", "2"), encoding="utf-8")

    # 5. One subsection (category) per top-level command, ordered alphabetically.
    for position, top in enumerate(sorted(groups), start=position_base):
        write_category_json(dst_section / top, top, position)

    # 6. Drop any now-empty source subfolders (e.g. the original flat `commands/`).
    for child in list(dst_section.iterdir()):
        if child.is_dir() and child.name not in groups and not any(child.rglob("*")):
            shutil.rmtree(child)

    # 7. Recompute every `.md` link against the new layout, keyed by basename stem.
    def relink(text: str, page_dir: Path) -> str:
        def repl(match: re.Match[str]) -> str:
            target = match.group(2)
            anchor = match.group(3) or ""
            base = target.split("/")[-1]
            if base not in stem_to_target:
                return match.group(0)
            rel = os.path.relpath(dst_section / stem_to_target[base], page_dir).replace(os.sep, "/")
            if not rel.startswith((".", "/")):
                rel = "./" + rel
            return f"{match.group(1)}{rel}{anchor}{match.group(4)}"
        return MD_LINK_RE.sub(repl, text)

    for path in dst_section.rglob("*.md"):
        path.write_text(relink(path.read_text(encoding="utf-8"), path.parent), encoding="utf-8")


def parse_source_index(index_text: str) -> Dict[str, Dict[str, str]]:
    """Extract per-module summaries from a source's flat index.md.

    Returns {file_stem: {"name": ..., "counts": ..., "summary": ...}}.
    """
    modules: Dict[str, Dict[str, str]] = {}
    for line in index_text.splitlines():
        match = SOURCE_INDEX_ENTRY_RE.match(line)
        if not match:
            continue
        modules[match.group("file")] = {
            "name": match.group("name"),
            "counts": match.group("counts") or "",
            "summary": (match.group("summary") or "").strip(),
        }
    return modules


def render_landing_page(landing: Dict, modules: Dict[str, Dict[str, str]], title: str) -> str:
    """Render a grouped overview page from `landing` config + parsed module metadata."""
    lines: List[str] = ["---", f"title: {json.dumps(landing.get('title', title))}", "sidebar_position: 1", "---", ""]
    lines.append(f"# {landing.get('title', title)}")
    lines.append("")
    intro = landing.get("intro")
    if intro:
        lines.append(intro)
        lines.append("")

    def module_link(stem: str, meta: Dict[str, str], folder: str) -> str:
        target = f"./{folder}/{stem}" if folder else f"./{stem}"
        link = f"[`{meta['name']}`]({target})"
        tail_bits = []
        if meta["counts"]:
            tail_bits.append(f"_{meta['counts']}_")
        if meta["summary"]:
            tail_bits.append(meta["summary"])
        tail = " — " + " ".join(tail_bits) if tail_bits else ""
        return f"- {link}{tail}"

    listed: set = set()
    for group in landing.get("groups", []):
        folder = group.get("dir") or group_slug(group["label"])
        lines.append(f"## {group['label']}")
        lines.append("")
        group_intro = group.get("intro")
        if group_intro:
            lines.append(group_intro)
            lines.append("")
        for module_stem in group.get("modules", []):
            meta = modules.get(module_stem)
            if not meta:
                LOG.warning("landing page references unknown module stem: %s", module_stem)
                continue
            listed.add(module_stem)
            lines.append(module_link(module_stem, meta, folder))
        lines.append("")

    leftover = [stem for stem in modules if stem not in listed]
    if leftover:
        lines.append("## Other modules")
        lines.append("")
        for module_stem in leftover:
            lines.append(module_link(module_stem, modules[module_stem], ""))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def maybe_write_landing_page(source: Dict, src_docs: Path, dst_section: Path, title: str) -> None:
    """If `landing_page` is configured for `source`, overwrite dst_section/index.md."""
    landing = source.get("landing_page")
    if not landing:
        return
    src_index = src_docs / "index.md"
    if not src_index.is_file():
        LOG.warning("[%s] landing_page configured but no source index.md to parse", source["key"])
        return
    modules = parse_source_index(src_index.read_text(encoding="utf-8"))
    excluded = set(source.get("exclude_files", []) or [])
    if excluded:
        modules = {stem: meta for stem, meta in modules.items() if stem not in excluded}
    landing_md = render_landing_page(landing, modules, title)
    (dst_section / "index.md").write_text(landing_md, encoding="utf-8")
    LOG.info("[%s] wrote grouped landing page (%d modules across %d groups)",
             source["key"], len(modules), len(landing.get("groups", [])))


def process_source(source: Dict, repo_root: Path, build_dir: Path, out_root: Path) -> Tuple[bool, str]:
    key = source["key"]
    repo = source["repo"]
    branches, snap_target = resolve_branch_candidates(source, repo_root)
    githash = source.get("githash", "") or ""
    docs_subpath = source.get("docs_subpath", "docs")
    mount = source.get("mount", key)
    title = source.get("title", key)
    sidebar_position = int(source.get("sidebar_position", 99))
    files_order = source.get("files_order", []) or []
    exclude_files = source.get("exclude_files", []) or []
    restructure_api = bool(source.get("restructure_api", False))
    link_rewrites = normalize_link_rewrites(source.get("link_rewrites", []) or [])

    staging = build_dir / "autodoc" / key
    LOG.info("[%s] clone/update %s (candidates: %s)", key, repo, githash or ", ".join(branches))
    try:
        used_branch = acquire_source(repo, branches, githash, staging)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip().splitlines()
        reason = stderr[-1] if stderr else f"exit {exc.returncode}"
        command = " ".join(str(part) for part in (exc.cmd or []))
        return False, f"git failed ({command}): {reason}"
    except FileNotFoundError:
        return False, "git binary not found"

    if githash:
        branch_reason = "githash"
    elif snap_target is None:
        branch_reason = "configured"
    elif used_branch == snap_target:
        branch_reason = f"snap: matched core@{used_branch}"
    else:
        branch_reason = f"snap: fell back to {used_branch}"
    LOG.info("[%s] using %s @ %s (%s)", key, repo, githash or used_branch, branch_reason)

    src_docs = staging / docs_subpath
    if not src_docs.is_dir():
        return False, f"docs_subpath '{docs_subpath}' not found in clone"

    dst_section = out_root / mount
    landing = source.get("landing_page")
    group_index = build_group_index(landing)
    LOG.info("[%s] copy %s -> %s", key, src_docs, dst_section)
    try:
        file_count = copy_section(
            src_docs,
            dst_section,
            files_order,
            group_index,
            exclude_files,
            restructure_api,
            link_rewrites,
        )
        write_category_json(dst_section, title, sidebar_position)
        write_group_categories(dst_section, landing)
        maybe_write_landing_page(source, src_docs, dst_section, title)
        group_commands = source.get("group_commands")
        if group_commands:
            regroup_command_pages(dst_section, group_commands)
    except OSError as exc:
        return False, f"copy failed: {exc}"

    return True, f"staged {file_count} files into {dst_section.relative_to(repo_root)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull external repo docs into core's docs tree.")
    parser.add_argument("--conf", required=True, help="Path to autodoc manifest JSON.")
    parser.add_argument("--repo-root", required=True, help="Core repo root.")
    parser.add_argument("--build-dir", required=True, help="Build directory (relative to repo-root or absolute).")
    parser.add_argument("--out-root", required=True, help="Where staged docs sections land (typically <repo-root>/docs).")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[autodoc] %(message)s",
    )

    repo_root = Path(args.repo_root).resolve()
    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = (repo_root / build_dir).resolve()
    out_root = Path(args.out_root).resolve()

    manifest = json.loads(Path(args.conf).read_text(encoding="utf-8"))
    sources = manifest.get("sources", [])
    if not sources:
        LOG.info("manifest has no sources; nothing to do")
        return 0

    failures: List[str] = []
    successes: List[str] = []
    for source in sources:
        ok, message = process_source(source, repo_root, build_dir, out_root)
        if ok:
            successes.append(f"{source['key']}: {message}")
            LOG.info("[%s] OK: %s", source["key"], message)
        else:
            failures.append(f"{source['key']}: {message}")
            LOG.warning("[%s] SKIPPED: %s", source["key"], message)

    LOG.info("summary: %d ok, %d skipped", len(successes), len(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
