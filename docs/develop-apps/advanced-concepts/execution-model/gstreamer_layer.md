---
title: GStreamer underneath
description: What the framework abstracts over GStreamer, what bleeds through, and when to reach for raw GStreamer.
sidebar_position: 3
slug: /develop-apps/advanced-concepts/gstreamer_layer
---

# GStreamer underneath

The Neat framework's pipelines run on GStreamer. Almost all of what's visible to the application — `Graph`, `Run`, `Node` — is a typed wrapper over GStreamer concepts. This page explains the layering: what gets hidden, what doesn't, and when raw GStreamer is the right answer.

## What the framework abstracts

| GStreamer concept | Framework abstraction |
|---|---|
| `gst-launch` text fragment | `Node::backend_fragment(int node_index)` |
| Element name (`name=…`) | Deterministic `n<idx>_<role>` from `Node::element_names()` |
| Pipeline string (concatenated fragments) | `Graph::add()` builds and concatenates |
| Caps negotiation | `Graph::build()` validates caps via `NodeCapsBehavior` |
| `gst_pipeline_set_state()` | `Graph::run()` / `Run::start()` |
| Bus messages | `GraphReport::bus_messages` |
| `appsrc` push API | `Run::push()` (only on Nodes with `InputRole::Push`) |
| `appsink` pull API | `Run::pull()` |
| Per-element timing | `RunDiagSnapshot::element_timing` |

The application never writes a launch string directly, never names elements directly, and never touches the GStreamer C API. Everything goes through Nodes.

## What bleeds through

A few GStreamer concepts the framework doesn't (and shouldn't) hide:

- **Caps semantics** — what fields a video / audio cap carries. Application code can read [`FormatTag`](/reference/cppapi/files/include-pipeline-formatspec-h) and inspect `Sample` metadata, which mirrors the relevant cap fields.
- **Buffer flags** — discontinuity, EOS, gap. The framework propagates these on `Sample` so application code can react to stream boundaries.
- **Event ordering** — GStreamer guarantees that events (caps, segment, EOS) flow in-order with buffers. The framework preserves this on the pull side.

If you need to know the exact GStreamer launch string for a built Graph, call `Graph::describe()` — it produces the deterministic `gst-launch` reproducer that recreates the pipeline byte-for-byte.

## When to reach for raw GStreamer

You don't need to in normal application code. The cases where it's appropriate:

- **Custom GStreamer plugins** — if you want a GStreamer element the framework doesn't ship as a Node, write a Node subclass that wraps your plugin and emits the right `backend_fragment()`. See "Building a custom Node" in the design deep dive (§0.10).
- **Diagnostic tooling** — the `repro_gst_launch` reproducer from `Graph::describe()` is exactly the launch string GStreamer would consume; you can paste it into `gst-launch-1.0` for offline debugging.
- **Plugin authoring** — SiMa's own GStreamer plugins (the `sima*` family) are documented in the plugin manifest ABI (see [`gst/SimaPluginStaticManifestAbi.h`](/reference/cppapi/files/include-gst-simapluginstaticmanifestabi-h)) and are loaded by the framework automatically.

## Determinism guarantees

The framework's element naming is deterministic — the same Node list with the same options always produces the same `gst-launch` string. This is what makes:

- The `repro_gst_launch` field actually reproducible.
- Test snapshots stable across runs.
- Element identification (e.g., for `RunDiagSnapshot` lookups) machine-friendly.

The convention is `n<node_index>_<role>`, where `role` is a short stable identifier the Node author picks. See the [Node API group](/reference/cppapi/groups/nodes) for the public Node wrappers that participate in this naming scheme.

## Further reading

- "GStreamer abstraction" — §0.8 of the design deep dive.
- [Node APIs](/reference/cppapi/groups/nodes) — concrete Node wrappers that emit deterministic backend fragments.
- [`Graph::describe()`](/reference/cppapi/classes/simaai-neat-graph) — print the launch string.
- "SiMa plugin manifest" — §51 and §95 of the design deep dive.
