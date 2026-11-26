---
name: core-update-docs
description: Edit Core guides, API reference source, examples, or tutorials. Use for docs and tutorial changes that need checked commands, translations, and source hashes.
---

# Update Core docs

## Workflow

1. **Scope the edit.** State what the reader needs to know or do and where
   the current guidance falls short. Follow the
   [documentation policy](../../../docs/develop-apps/contribute/start-here/coding_standard.md#documentation-obligations)
   and update the smallest set of pages that addresses the need.
2. **Edit the owning source.** Edit authored pages under `docs/`, tutorial
   READMEs under `tutorials/` or `pcie_host/tutorials/`, and public API
   comments under `include/`. Edit autodoc imports in their source repo.
   Keep the instructions focused on the reader's task. Put implementation
   rationale and investigation details in the PR.
3. **Update translations.** Check the [localization scope](../../../website/i18n/scope.json)
   and [localization guide](../../../website/i18n/README.md). Tutorial
   translations live beside their source READMEs. Follow the
   [tutorial localization guide](../../../tutorials/i18n/README.md).
   Update the required locale pages and matching source hashes in the same
   PR. AI can draft translations. Verify meaning, commands, symbols, links,
   and protected markup against the source before refreshing hashes. Follow
   the technical review requirements in the guides.
4. **Verify and report.** Check commands, API names, and prerequisites against
   the current source or installed package. Run changed commands or examples
   in the documented environment when applicable. From the repo root, run
   `bash scripts/ci/check_documentation_localization.sh` and
   `DOCS_STRICT_LINKS=1 ./build.sh --doc`. Ask the contributor to read the
   changed docs. Link the changed sections and name the instructions, claims,
   or expected results to verify against supported behavior. Get their
   confirmation before marking the PR ready. Record the commands, tested
   environment, results, and any outstanding checks in the PR.
