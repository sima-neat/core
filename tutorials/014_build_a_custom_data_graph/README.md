# 014 Build a Custom Data Graph

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Labels | graph, traversal, metadata |

## Concept

Build the smallest useful NEAT graph — one pipeline node wired to one stage node — then push a sample through and verify metadata survives traversal. This is the baseline before hybrid or multistream graph tutorials.

A graph is an explicit DAG of nodes you build programmatically, separate from the pipeline/session abstraction. You add nodes with `graph.add(...)`, wire them with `graph.connect(...)`, and run the whole thing via `GraphSession`.

**APIs introduced**
- `pyneat.graph.Graph()` — the graph container.
- `graph.add(node)` — add a node; returns an ID.
- `graph.connect(src_id, dst_id)` — wire outputs to inputs.
- `pyneat.graph.nodes.pipeline_node(inner_node, name)` — wrap a regular pipeline node for use inside a graph.
- `pyneat.graph.nodes.stamp_frame_id(name)` — a stage node that tags samples with frame identity.
- `pyneat.graph.GraphSession(graph).build()` — materialize the graph into a runnable.

**When to use this**
- Custom orchestration where pipeline/session semantics don't fit (fan-out, fan-in, per-stream routing).
- Multistream scheduling (see chapter 016).
- Embedding model execution as one stage of a larger flow (see chapter 015).

**Prerequisites**
Chapter 003 (Session basics).

**References**
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

**Python:**
```bash
python3 share/sima-neat/tutorials/014_build_a_custom_data_graph/build_a_custom_data_graph.py
```

**C++:**
```bash
./lib/sima-neat/tutorials/tutorial_v2_014_build_a_custom_data_graph
```

To compile this chapter's C++ source in your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials/v2#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/014_build_a_custom_data_graph/build_a_custom_data_graph.cpp`
- Python: `tutorials/014_build_a_custom_data_graph/build_a_custom_data_graph.py`
