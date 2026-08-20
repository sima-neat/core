# Documentation localization

The deployed Developer Center shell owns the visible documentation-language
selector and persists the choice across Hardware and Software. This repository
does not render a separate locale dropdown. The legacy `?i18n=1` URL remains
available for editorial review links.

For a standalone local Software docs server, set the preferred locale before
starting Docusaurus:

```bash
DOCS_PREFERRED_LOCALE=ja npm start
```

The config exposes that value as `window.__NEAT_DOCS_PREFERRED_LOCALE__`, which
may also be injected by a local host page before the docs bundle loads. Valid
values are `en`, `ko`, `ja`, `zh-Hant`, and `uk`.

The initial review slice covered installation, the Neat Library landing page,
and the minimal C++/Python validation application. Translation is now expanding
to the maintained narrative documentation. `scope.json` is the authoritative
boundary: reference and Doxygen-derived pages remain English-only during the
preview because much of that surface is generated. The original pilot page list
and English source hashes remain in `pilot-translations.json`.

## Translation conventions

- Keep product names (`Palette Neat`, `Neat SDK`, `Neat Library`, `Modalix`,
  `DevKit`, `PyNeat`, `Insight`, and `LLiMa`) in English.
- Keep API symbols, commands, paths, environment variables, filenames, literal
  output, and error messages unchanged.
- Keep localized documentation images as base-aware `@site/../docs/...`
  imports. Do not use `pathname:///`; it bypasses the `/software/` deployment
  base path in the Developer Center.
- Translate conceptual nouns naturally, but introduce the English term when it
  improves searchability or removes ambiguity.
- Korean uses a consistent formal instructional style.
- Japanese uses a concise `です・ます` instructional style.
- Traditional Chinese targets Taiwan (`zh-Hant-TW`) terminology.
- Ukrainian targets Ukraine (`uk-UA`) terminology and spelling, using
  `застосунок` for a software application and `тека` for a folder.
- Do not treat an AI draft as publication-ready until a native technical
  reviewer approves it.
- Follow the canonical term choices in `terminology.md`.

Run `npm run check:i18n` during translation work. It reports authored-page
coverage, rejects changes to protected structure in every translation that
exists, and flags sidebar categories left in English unless the label is a
canonical product, command, or API name. `translation-sources.json` records the
English source hash reviewed by each locale, so the check also rejects stale or
untracked translations. Run
`npm run check:i18n-complete` as the publication gate; it additionally fails if
any tracked authored page lacks any locale. The original
`npm run check:i18n-pilot` check remains available for source-hash validation of
the initial review slice.

## Remote Ollama drafting

`scripts/translate-docs-ollama.js` sends protected prose blocks to an Ollama
server through SSH. Code, commands, inline identifiers, links, JSX/HTML tags,
and canonical product names are replaced with verified placeholders before the
request. The script aborts if the model changes or drops any placeholder.

Preview one page without writing files:

```bash
npm run translate:i18n -- \
  --locale zh-Hant \
  --source docs/tools/index.md \
  --ssh-host macstudio \
  --model translategemma:27b
```

Add `--write` only after inspecting representative output. Use `--all --write`
to fill all missing pages for one locale, or `--prefix docs/release-notes/` to
process one section. Existing translations are skipped unless `--overwrite` is
also supplied. After each batch, run `npm run check:i18n` and the relevant
Docusaurus locale build before editorial review.

Traditional Chinese output also receives a protected Taiwan terminology pass.
It changes prose only; fenced code, `ShellCommand` blocks, and inline code are
left byte-for-byte unchanged. Reapply the current terminology rules to existing
pages without contacting Ollama:

```bash
npm run translate:i18n -- \
  --locale zh-Hant \
  --all \
  --write \
  --normalize-existing
```

## Generated and reference documentation

Generated documentation is deliberately outside the authored-content
translation workflow. `scope.json` is the authoritative boundary. It includes
reference and Doxygen output as well as the Model Compiler, LLiMa, Insight,
Sentinel, and sima-cli families imported by autodoc. Those routes remain
English-only. Generated tutorial routes are also English-only except for the
authored `/tutorials/before-you-run` page. The site labels these boundaries in
localized navigation. Do not hand-edit localized copies of generated output in
this repository; the next generator run would make those edits stale or
overwrite them.

If a generated documentation family is localized later, localization belongs
with the source repository or generator that owns its schema and templates. The
producer should emit stable page and symbol identifiers plus locale-specific
titles, descriptions, explanatory prose, and navigation metadata. API symbols,
signatures, code, and machine-readable identifiers remain canonical. Core should
then consume the generated locale artifacts reproducibly, record their source
version or hash, and validate route parity before removing that family from
`scope.json`.

Onboarding a generated family therefore requires all of the following:

1. a reproducible localized generation command in the owning repository;
2. the shared terminology rules in `terminology.md` (or a synchronized copy);
3. stable locale output paths and source-version metadata;
4. structural and link validation equivalent to `check:i18n`; and
5. removal of the exact generated prefix from `scope.json` only after all four
   locales build successfully.

This keeps generated translations source-owned and updateable without forcing
each downstream repository to maintain hand-translated Doxygen output.
