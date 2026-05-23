# 014 Run Multiple Streams in One Graph

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | None |
| Labels | graph, multistream, scheduler, join |

## Concept

Run multiple logical streams through one public `Graph` and combine two named inputs into one deterministic bundle output. This is the pattern behind multi-camera or multi-source systems where related inputs must be aligned before downstream processing.

The graph here is created with:

```python
graph = pyneat.graphs.combine(["left", "right"],
                              "combined",
                              pyneat.CombinePolicy.ByFrame)
```

Each pushed sample carries a `stream_id` and `frame_id`. `CombinePolicy.ByFrame` waits until both named inputs have produced the same `frame_id`, then emits one bundle from `run.pull("combined")`.

**APIs introduced**
- `pyneat.graphs.combine(inputs, output, policy)` — build a reusable public Graph fragment.
- `pyneat.CombinePolicy.ByFrame` — combine samples whose `Sample.frame_id` values match.
- `pyneat.CombinePolicy.ByPts` — combine samples whose `Sample.pts_ns` presentation timestamps match.
- `run.push("left", [sample])` / `run.push("right", [sample])` — named multi-input push.
- `run.pull("combined")` — named output pull.

**When to use this**
- Multi-camera ingestion where each stream must make progress independently.
- Parallel branch processing (e.g. two models running side-by-side) that must rejoin outputs correctly.
- Diagnosing dropped or misaligned stream outputs under load.

**Prerequisites**
Chapter 012 (Graph basics). Chapter 009 (bundle samples) helps for join semantics.

**References**
- [Graph](/getting-started/programming-model/graph)
- [Pipeline](/getting-started/programming-model/pipeline)

## Learning Process
1. Generate deterministic per-stream/per-frame samples with explicit tags.
2. Build a public combine Graph with named inputs and one named output.
3. Push all expected inputs and pull joined outputs.
4. Validate output count and bundle cardinality.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/014_run_multiple_streams/run_multiple_streams.py \
  --streams 8 --frames 4
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_014_run_multiple_streams \
  --streams 8 --frames 4
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_014_run_multiple_streams
./build/tutorials-standalone/tutorial_014_run_multiple_streams \
  --streams 8 --frames 4
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/014_run_multiple_streams/run_multiple_streams.cpp`
- Python: `tutorials/014_run_multiple_streams/run_multiple_streams.py`
