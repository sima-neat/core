# Rolling documentation releases

`current.json` is the committed documentation release snapshot (schema version 1).
Each scalar has a `value`, an accountable `owner`, and a `source` explaining its
provenance. Values have no `v` prefix. Core and platform must exactly match
`package-version` and `platform-version` in `deps/manifest.json`. The SDK release
channel is the published installation channel, **not** `deps/manifest.json`'s
build dependency channel. Other product owners supply their supported release
versions and validate the combinations before changing this snapshot.

Use `%%release.core%%`, `%%release.platform%%`, `%%release.sdk%%`,
`%%release.sdk_channel%%`, `%%release.sdk_series%%`, `%%release.apps%%`,
`%%release.model_compiler%%`, `%%release.insight%%`, and `%%release.sima_cli%%`
only for rolling/current values. Docusaurus substitutes them before Markdown/MDX
parsing, including prose, inline code, fenced commands, links, tables, and
`ShellCommand` children. No network request or client-side resolution is used.
The validated snapshot is embedded in the preprocessor's cache identity, so a
release-data-only edit invalidates compiled pages. Restart a running dev server
after changing release data.

The Algolia record generator also uses the shared resolver, through
`website/scripts/resolve-release-sources.js`, before extracting search text.
It batches token-bearing source documents per locale and fails on invalid tokens
or release metadata, so search records contain the same concrete versions as the site.

The compatibility page and its translations are maintained manually, with concrete
versions throughout its prose, commands, tables, links, and CLI examples.
`policy.json` designates this page in `manualPaths`; validation rejects release
tokens there. Review and update supported version combinations separately during
each release. Updating release data never changes the compatibility page.

Historical release notes, firmware-family examples, minimum-version requirements,
and version-specific migration instructions also stay literal. `policy.json`
designates rolling pages and exact-line, counted exceptions with reasons.
Exceptions include network addresses, which resemble versions. Do not allowlist
a whole directory or a version globally.

For a release, update `current.json` once after the authoritative package manifest
has been updated, then run:

```bash
npm --prefix website run sync:release-readmes
npm --prefix website run check:releases
./build.sh --doc
```

`sync:release-readmes` only updates the marked command block in
`pcie_host/README.md`, from the template in `policy.json`. Commit this concrete
GitHub-readable output. Normal validation never rewrites it.

Keep tokens identical in all translations. The one-time migration updates source
hashes after verifying translated prose alignment; subsequent release-data edits
leave Markdown sources and translation hashes untouched. `check:releases`
rejects unknown/malformed tokens, token-set or occurrence-count mismatches, manifest drift, stale
README blocks, and unapproved literal version pins. It checks pins of *any*
version so a bump cannot hide an old stale pin. `npm run build` runs these checks,
the existing complete localization/catalog/navigation checks, and tests. Its
Docusaurus postBuild hook scans all HTML for unresolved tokens and verifies representative
commands in every locale. The hook uses each locale's actual output
root, so `--locale` and `--out-dir` builds receive the same checks.

CLI sample layout outside the manual compatibility page remains authored Markdown.
