# 014 Graph Basics

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Labels | graph, traversal, metadata |

## Concept
This tutorial introduces the Neat graph runtime using the smallest useful graph: push one sample through connected nodes and verify metadata survives traversal.

If you are new to graph APIs, read this chapter before hybrid/multistream graph tutorials. It teaches the baseline mental model:
- Build graph nodes and edges.
- Start a `GraphSession`, then push/pull samples.
- Validate stream/frame metadata after graph execution.

What this chapter demonstrates:
- Preferred hybrid path (pipeline node + stage node).
- Deterministic fallback path (stage-only graph).
- Signature checks that confirm graph output contracts.

Reference:
- [Graph](/getting-started/programming-model/graph)
- [Session](/getting-started/programming-model/session)

## Learning Process
1. Build a minimal graph and push one deterministic tensor sample.
2. Run preferred pipeline+stage composition, with stage-only fallback when needed.
3. Pull graph output and validate stream/frame metadata stamping.
4. Use `CHECK`, `SIGNATURE`, and `[OK]` markers to confirm graph correctness.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_014_graph_basics
python3 tutorials/014_graph_basics/graph_basics.py
```

## Source Files
- C++: `tutorials/014_graph_basics/graph_basics.cpp`
- Python: `tutorials/014_graph_basics/graph_basics.py`
