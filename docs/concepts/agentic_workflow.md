---
title: Agentic workflow
description: Why the framework's API surface is designed for AI agents that write Modalix code.
sidebar_position: 9
---

# Agentic workflow

The Neat framework's public API is designed not just for human application authors, but for AI coding agents that write Modalix-targeted code on a developer's behalf. This page explains why several of the framework's design choices look the way they do — they're optimized for the agent reading the API surface, not just the human.

## What an agent needs from a framework

An AI agent generating code against an unfamiliar API typically needs:

1. **A small public surface** — fewer types means less to learn.
2. **Self-describing types** — the type signature should encode the contract, not just hint at it.
3. **One way to do each thing** — multiple equivalent paths are exactly the kind of choice an agent gets wrong.
4. **Adjacent docs** — the agent reads `@brief` blocks, not 8000-line design docs. Critical context must live next to the symbol.
5. **Deterministic outputs** — when an agent writes test snapshots, byte-for-byte determinism makes them stable.
6. **Errors that name root causes** — when something fails, the error needs to say which kernel / contract / option is wrong, not "session build failed."
7. **Reproducer artifacts** — if the agent's code fails, it can paste the `repro_gst_launch` string into a debugger and iterate without rerunning the agent.
8. **No hidden globals** — every behavior should be a constructor argument or a Node option, not an environment variable read at runtime.
9. **Stable identifiers** — same Nodes + same options ⇒ same element names ⇒ same launch string ⇒ stable test fixtures.
10. **Model packs as data, not code** — the model is a bundle the framework loads, not a header the agent has to compile against.
11. **Composable building blocks** — Nodes and NodeGroups snap together; the agent doesn't have to know which combinations the framework "supports" because the planner answers that at build time.
12. **Loud build-time validation** — failures surface from `Session::build()`, not at the first `pull()`. An agent's iteration loop needs the error pointed at the right line.
13. **Structured `SessionReport`** — programmatic access to what the build did, so the agent can verify its own work.
14. **Progressive disclosure** — `Model` is the simple path, `Session::add()` is the next, custom Nodes are the deep path. An agent hits the simple path first.
15. **Versioned MPK contracts** — a model pack carries enough metadata that an agent generating code against it doesn't need to consult external docs.

These show up across the framework as concrete design decisions:

- The 8 main concepts (Model, Tensor, Node, NodeGroup, Session, Run, Builder Graph, Runtime Graph) are the entire public surface an agent needs to learn.
- `Session::describe()` returns a structured plan summary — agents can verify their own pipelines.
- `SessionError` carries `error_code()` plus a `SessionReport` — agents can switch on the code.
- `Node::backend_fragment()` is deterministic — agents can write golden-string tests against pipelines.
- `MpKManifest` is the only authoritative description of a model — agents generating wiring code against a model never need anything else.

## How application authors benefit

Even if you're not an agent, you benefit from agent-friendly API design: the same properties that make automated code generation reliable also make manual code review easier, refactoring safer, and AI-assisted development (Claude Code, Copilot, Cursor) more productive on Modalix code.

## Further reading

- "Introducing Neat" — §0.1 of the design deep dive.
- [`Session::describe()`](/reference/cppapi/classes/simaai-neat-session) — programmatic plan summary.
- [`SessionError`](/reference/cppapi/classes/simaai-neat-sessionerror) — structured error type.
- [`MpKManifest`](/reference/cppapi/structs/simaai-neat-mpk-mpkmanifest) — model-pack metadata.
