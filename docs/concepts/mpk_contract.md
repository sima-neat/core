---
title: MPK contract
description: How the MPK inference contract relates to .tar.gz model archives.
sidebar_position: 12
---

# MPK contract

**MPK** is the inference contract embedded in a compiled model archive. It is a JSON
file named `mpk.json` or `*_mpk.json` that the route planner and downstream extractors
read. Model archives themselves are ordinary `.tar.gz` files loaded through `Model`.

This page separates the two concerns:

- **The archive** — a `.tar.gz` distribution unit containing binaries, configs, weights,
  and the MPK contract JSON.
- **The contract** — the `*_mpk.json` JSON document that is the source of truth for
  inference routing.

For byte-level contract and security rules, see
[MPK Contract (contributor reference)](/contribute/mpk_contract).

## Why a single `.tar.gz` archive

Models on Modalix have many moving parts (compiled MLA graph, CVU kernels, configs,
weights). Bundling them into one signed `.tar.gz` archive means:

- **Atomicity** — either the whole model loads or none of it does.
- **Versioning** — the MPK contract declares its schema version; incompatible contracts
  are rejected early.
- **Provenance** — one artifact to checksum, sign, and ship.
- **Reproducibility** — given the same archive and MPK contract, the framework's planner
  makes the same routing decisions.

## The MPK contract is the only authoritative JSON

A model archive may contain other JSON files; the framework treats only `mpk.json` or
`*_mpk.json` as the inference contract. Don't infer model topology from other JSON in
the archive — the contract is "if it's not in the MPK contract, it doesn't exist."

This rule keeps the framework's planner deterministic.

## Loading model archives

Applications do not call an archive-loader API directly. Construct `Model` with the
`.tar.gz` path; `Model` performs archive validation, extracts the safe contents into an
internal layout, parses the MPK contract, and runs route planning.

Only exact lowercase `.tar.gz` model archives are accepted. `.mpk`, `.tgz`, `.tar`, and
bare `.gz` files are rejected before archive inspection.

## Security defenses

Because model archives come from outside the system, loading is **fail-closed**. Every
accepted archive passes a strict matrix:

1. **Extension allowlist** — only `.tar.gz` is accepted.
2. **Size caps** — archive, entry, and total JSON byte limits.
3. **Entry-count caps** — protects against zip-bomb-style packs.
4. **JSON depth cap** — protects against parser-stack exhaustion.
5. **Path traversal protection** — normalized paths reject absolutes and `..` segments.
6. **UTF-8 validation** — entry paths must be valid UTF-8.
7. **Unicode confusable rejection** — visually-confusable path characters are rejected.
8. **File-type allowlist** — only known file types extract.
9. **JSON duplicate-key rejection** — duplicate keys at any depth are rejected.
10. **Schema validation** — contract and loader-side metadata must match supported schemas.
11. **Required-section validation** — the inference contract and model binary must be present.
12. **Kernel allowlist** — referenced kernels must be in the framework's known set.

Higher layers report failures as `NeatError` with structured `io.*` or `mpk.*` error
codes.

## Related types

- [`Model`](/reference/cppapi/classes/simaai-neat-model) — the public entry point that
  loads `.tar.gz` model archives and exposes route fragments.
- `MpkContract` and related internal structs — the parsed inference contract from
  `mpk.json` / `*_mpk.json`.

## Further reading

- [MPK Contract (contributor reference)](/contribute/mpk_contract) — byte-level rules and field semantics.
- "MPK contract" — §0.16, §15 of the design deep dive.
- "Model archive security matrix" — §91 of the design deep dive.
