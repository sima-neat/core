# 016 Run Multiple Streams in One Graph

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Labels | graph, multistream, scheduler, join |

## Concept

Schedule multiple streams through one graph — fair scheduling, fan-out branches, and deterministic bundle re-join. This is the pattern behind any multi-camera or multi-source system built on NEAT graphs.

The graph here is: `stamp → stream_scheduler → fan_out → join_bundle`. Each sample is tagged with stream/frame identity, scheduled fairly across streams, branched into parallel paths, and re-joined into a bundle you can pull as one unit.

**APIs introduced**
- `pyneat.graph.nodes.StreamSchedulerOptions()` with `per_stream_queue`, `drop_policy`.
- `pyneat.graph.nodes.stream_scheduler(opts, name)` — the fairness primitive.
- `pyneat.graph.nodes.fan_out(port_names, name)` — split one sample into multiple named branches.
- `pyneat.graph.nodes.join_bundle(port_names, name, bundle_name)` — re-join branches into one bundle.
- `pyneat.graph.nodes.StreamDropPolicy.DropOldest` — per-stream overflow policy.

**When to use this**
- Multi-camera ingestion where each stream must make progress independently.
- Parallel branch processing (e.g. two models running side-by-side) that must rejoin outputs correctly.
- Diagnosing dropped or misaligned stream outputs under load.

**Prerequisites**
Chapter 014 (Graph basics). Chapter 009 (bundle samples) helps for join semantics.

**References**
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
./tutorial_v2_016_run_multiple_streams
python3 tutorials/016_run_multiple_streams/run_multiple_streams.py
```

## Source Files
- C++: `tutorials/016_run_multiple_streams/run_multiple_streams.cpp`
- Python: `tutorials/016_run_multiple_streams/run_multiple_streams.py`
