---
title: MPK contract
description: What an MPK is, what's inside, and the security defenses applied to every load.
sidebar_position: 12
---

# MPK contract

An **MPK** (Model Pack) is a `.tar.gz` / `.tgz` / `.mpk` / `.tar` archive containing everything the framework needs to run a model on Modalix:

- A **manifest** (`mpk.json`) — the authoritative description of the pack.
- A **model binary** — `.lm`, `.so`, or equivalent, depending on the target.
- **Configs** — per-stage runtime metadata.
- **Weights / assets** — quantization tables, calibration data, anything the model needs at runtime.

This page is the conceptual overview. For the byte-level contract and security rules, see [MPK Contract (contributor reference)](../contribute/mpk_contract).

## Why a single archive

Models on Modalix have many moving parts (compiled MLA graph, CVU kernels, configs, weights). Bundling them into one signed archive means:

- **Atomicity** — either the whole pack loads or none of it does.
- **Versioning** — the manifest declares its schema version; old loaders reject incompatible packs early.
- **Provenance** — one artifact to checksum, sign, and ship.
- **Reproducibility** — given the same MPK, the framework's planner makes the same routing decisions.

## The manifest is the only authoritative JSON

An MPK may contain other JSON files; the framework reads only `mpk.json` (or `*_mpk.json`). Don't infer anything about the model from other JSON in the archive — the contract is "if it's not in the manifest, it doesn't exist."

This is enforced by the loader and is the rule that lets the framework's planner be deterministic.

## Inspect-only vs full extraction

`MpKLoader::inspect()` reads and validates the manifest without touching the rest of the archive. Useful when tooling just wants to inspect a model's metadata (preprocess shape, op set, version).

`MpKLoader::extract()` runs the full validation matrix and unpacks everything. Required before a `Model` can run.

## Security defenses

Because MPKs come from outside the system, the loader is **fail-closed**: every accepted pack passes a strict matrix:

1. **Size caps** — `max_archive_bytes`, `max_entry_bytes`, `max_total_json_bytes`.
2. **Entry-count caps** — `max_entries` — protects against zip-bomb-style packs.
3. **JSON depth cap** — `max_json_depth` — protects against parser-stack exhaustion.
4. **Path traversal protection** — every entry's path is normalized; absolute paths and `..` segments are rejected.
5. **UTF-8 validation** — entry paths must be valid UTF-8.
6. **Unicode confusable rejection** — visually-confusable characters in paths are rejected (typosquatting defense).
7. **File-type allowlist** — only known file types extract.
8. **JSON duplicate-key rejection** — manifests with duplicate keys at any depth are rejected.
9. **Schema validation** — manifest must match the loader's schema for its declared version.
10. **Required-section validation** — `pipeline_sequence` and at least one model binary must be present.
11. **Kernel allowlist** — referenced kernels must be in the framework's known set.

… and more. The full 25-defense matrix is in §91 of the design deep dive.

A failure in any of these raises `MpKError` with the matching `ErrorClass` (`InvalidArchive` / `PathTraversal` / `SchemaError` / `UnsupportedVersion` / `SizeLimitExceeded`). Higher layers wrap this into a `SessionError` with an `error_code` of the form `io.*` or `mpk.*`.

## Tightening the defaults

Default `MpKLoaderOptions` are conservative enough for production model packs. For sandboxed / multi-tenant deployments where untrusted packs flow through the loader, tighten:

```cpp
sima::MpKLoaderOptions opt;
opt.max_archive_bytes = 64ULL * 1024ULL * 1024ULL;   // 64 MiB instead of 512 MiB
opt.max_entries = 256;
opt.reject_unicode_path_confusables = true;          // already on, but be explicit
auto manifest = sima::MpKLoader::inspect(path, opt);
```

## Related types

- [`MpKLoader`](/reference/cppapi/classes/simaai-neat-mpk-mpkloader) — entry point.
- [`MpKLoaderOptions`](/reference/cppapi/structs/simaai-neat-mpk-mpkloaderoptions) — tunable safety caps.
- [`MpKManifest`](/reference/cppapi/structs/simaai-neat-mpk-mpkmanifest) — parsed manifest.
- [`MpKError`](/reference/cppapi/classes/simaai-neat-mpk-mpkerror) — exception with `ErrorClass`.
- [`ErrorClass`](/reference/cppapi/enums/) — coarse triage taxonomy.

## Further reading

- [MPK Contract (contributor reference)](../contribute/mpk_contract) — byte-level rules and field semantics.
- "MPK contract" — §0.16, §15 of the design deep dive.
- "MPK security matrix" — §91 of the design deep dive.
