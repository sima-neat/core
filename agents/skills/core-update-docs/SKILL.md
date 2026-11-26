---
name: core-update-docs
description: Edit Core guides, API reference source, examples, or tutorials. Use for docs and tutorial changes that need checked commands, translations, and source hashes.
---

# Update Core docs

1. Find the existing source that owns the guidance. Edit authored pages under
   `docs/`, tutorial READMEs under `tutorials/`, and public API comments under
   `include/`. Update the source repository for pages imported by autodoc.
2. Check commands, API names, examples, and prerequisites against the current
   source or installed package. Keep each edit focused on what the reader needs
   to do. Put investigation details and validation logs in the PR.
3. Use the [localization guide](../../../website/i18n/README.md) and
   [tutorial localization guide](../../../tutorials/i18n/README.md) to locate
   affected pages. Update the required locale pages and their matching source
   hashes in the same PR. AI can draft translations. Verify meaning, commands,
   symbols, links, and protected markup against the source before refreshing
   hashes. Follow the technical review requirements in the localization guide.
4. From the repo root, run `bash scripts/ci/check_documentation_localization.sh`
   and `DOCS_STRICT_LINKS=1 ./build.sh --doc` for the build and link checks.
   Record the results and any external input still needed.
