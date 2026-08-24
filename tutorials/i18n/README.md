# Tutorial localization

Tutorial translations are owned beside each tutorial instead of editing the
generated Docusaurus pages.

For every supported locale, add:

```text
tutorials/001_run_your_first_model/
└── i18n/
    └── ja/
        └── README.md
```

PCIe tutorials use the same layout below `pcie_host/tutorials/`.

Copy the English `README.md` as the starting point. Translate the title and
prose, but retain these level-two headings in English because the generator uses
them as schema keys: `Metadata`, `Concept`, `Learning Process`, `Walkthrough`,
`Run`, and `In Practice`. Retain walkthrough `{#step-...}` anchors, code spans,
commands, paths, links, and fenced code exactly. Metadata keys, `Category`,
`Difficulty`, and `Labels` are canonical; only `Estimated Read Time` may have a
localized value.

Repeated generated text is translated once in `<locale>.json`. Optional landing
copy belongs in `<locale>/heading.mm`. A locale is generated only when all
tutorial folders contain its localized README, so the published tutorial area
never silently mixes translated and English pages.

`translation-sources.json` records the SHA-256 of each canonical README at the
time its translation was reviewed. Update the corresponding hash whenever the
translation is brought up to date. The docs build rejects missing or stale hash
records before generating localized tutorial pages.

To create protected translation drafts with the same Ollama workflow used by
the rest of the site, run:

```bash
npm --prefix website run translate:i18n-tutorials -- \
  --locale ja \
  --local \
  --write
```

Omit `--local` to use the configured SSH host. The shared CLI reads
`sima-i18n.tutorials.json`, preserves tutorial schema headings, metadata,
commands, code, links, and walkthrough anchors, and writes each result into its
tutorial folder. Treat the generated files and source-hash entries as drafts
until a native technical reviewer approves them.
