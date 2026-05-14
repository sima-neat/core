---
title: Tools and scripts inventory
description: What lives in `core/scripts/` and `core/tools/`, and when to reach for each.
sidebar_position: 11
---

# Tools and scripts inventory

The framework ships two directories of helpers. This page is the map.

## `core/tools/` — doc and build helpers

| Script | Purpose |
|---|---|
| `generate_api_docs.sh` | Run doxygen2docusaurus over the Doxygen XML; emit Markdown for the C++ API reference site. Run after editing public headers. |
| `generate_python_api_docs.py` | Generate Python API reference Markdown from the `pyneat` module's docstrings. |
| `generate_tutorial_docs.py` | (Tutorials are being deprecated — this script is fading out.) |
| `postprocess_d2d_links.py` | Fix up doxygen2docusaurus link slugs after generation. Called automatically by `generate_api_docs.sh`. |
| `strip_empty_programlisting.py` | Workaround for empty `<programlisting>` elements that confuse doxygen2docusaurus. |
| `compute_version.sh` | Compute the framework's version string from git state. Used by CI and packaging. |
| `expand_code_tabs.py` | Expand multi-language tabs in tutorial sources. |
| `run_clean_env.sh` | Run a command inside a clean shell environment (no inherited `LD_*` / `PATH` weirdness). |
| `tutorial_quality_lint.py` / `tutorial_scorecard.py` | Lint tutorial Markdown / score it. (Deprecating with tutorials.) |

Typical flow when editing public headers:

```bash
cd core
doxygen docs/doxygen/Doxyfile      # regenerate XML
bash tools/generate_api_docs.sh    # regenerate Markdown
cd website && yarn start           # preview the site
```

## `core/scripts/` — repo-level checks and dev helpers

| Script | Purpose |
|---|---|
| `check_format.sh` | Run clang-format on the C++ tree; fail on diffs. |
| `check_cmake_format.sh` / `check_cmake_style.py` | Run cmake-format / lint on `CMakeLists.txt` files. |
| `check_duplicate_includes.{sh,py}` | Catch duplicate `#include` lines in headers. |
| `check_internal_headers.sh` | Verify the `core/src/pipeline/internal/sima/` reach-through tier respects the public/internal boundary. |
| `run_cpp_tidy.sh` | Run clang-tidy across the tree. |
| `route_refactor_validation.sh` | A targeted route-planner regression check (called by CI). |
| `install_neat_plugins.sh` | Install the framework's GStreamer plugins to the system plugin directory. |
| `install_codex_skill.sh` | Install the Codex CLI's NEAT skill (developer convenience). |
| `fix_devkit_runtime.sh` | Patch a fresh devkit's runtime libs / paths. |
| `sync_neatdecoder.sh` / `use_neatdecoder.sh` | Switch between bundled and external decoder builds. |

### `core/scripts/ci/`, `core/scripts/dev/`, `core/scripts/release/`

These subdirectories hold scripts that are owned by their respective workflows — CI runs the `ci/` set, developers run `dev/` ad-hoc, release engineering runs `release/`. Don't depend on them from application code.

## Running the doc generator from a clean checkout

```bash
sudo apt-get install -y doxygen   # if not installed
cd core
doxygen docs/doxygen/Doxyfile      # generates docs/doxygen/out/xml/
bash tools/generate_api_docs.sh    # populates docs/reference/cppapi/
python3 tools/generate_python_api_docs.py   # populates docs/reference/pythonapi/
cd website && yarn install && yarn start    # serve at http://localhost:3000/
```

## Further reading

- "Tools and scripts" — §55 of the design deep dive.
- The repo `core/AGENTS.md` has the contributor agreement on what tools must run pre-commit.
