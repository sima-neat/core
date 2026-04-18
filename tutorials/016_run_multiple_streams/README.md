# 016 Run Multiple Streams in One Graph

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Labels | graph, multistream, scheduler, join |

## Concept
This tutorial teaches multistream graph scheduling fundamentals: how multiple stream/frame inputs are fairly scheduled, branched, and re-joined into deterministic bundles.

Why read this before production graph scaling: it gives you a concrete pattern for validating stream fairness and bundle cardinality, which are common failure points in multistream systems.

What this chapter demonstrates:
- Tagging each sample with stream/frame identity.
- Building graph path: stamp -> scheduler -> fanout -> join.
- Verifying expected bundle field count and output count.

Use-case guidance:
- Multi-camera ingestion where each stream must make progress.
- Parallel branch processing that must rejoin outputs correctly.
- Diagnosing dropped/misaligned stream outputs under load.

Reference:
- [Graph](/getting-started/programming-model/graph)
- [Pipeline](/getting-started/programming-model/pipeline)

## Learning Process
1. Generate deterministic per-stream/per-frame samples with explicit tags.
2. Build multistream graph with scheduler, fanout, and join stages.
3. Push all expected inputs and pull joined outputs.
4. Validate output count and bundle cardinality via `CHECK` and `SIGNATURE`.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_016_graph_multistream
python3 tutorials/016_graph_multistream/graph_multistream.py
```

## Source Files
- C++: `tutorials/016_graph_multistream/graph_multistream.cpp`
- Python: `tutorials/016_graph_multistream/graph_multistream.py`
