# 012 Build a Custom Data Graph

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | None |
| Labels | graph, traversal, metadata |

## Concept

Build the smallest useful public Neat `Graph` — one named `Input` wired to one named `Output` — then push a sample through and verify metadata survives traversal. This is the baseline before hybrid or multistream graph tutorials.

A public `Graph` is the application composition surface. You add Nodes with `graph.add(...)`, wire named public endpoints with `graph.connect("input_name", "output_name")`, build once with `graph.build()`, then use named runtime calls:

```python
run.push("image", [sample])
sample = run.pull("out")
```

The old low-level `pyneat.graph` module has been removed. Application code should use `pyneat.Graph` and reusable public Graph fragments.

**APIs introduced**
- `pyneat.Graph()` — the public graph container.
- `pyneat.nodes.input("image")` — a named push endpoint.
- `pyneat.nodes.output("out")` — a named pull endpoint.
- `graph.add(node)` — add a Node or reusable Graph fragment.
- `graph.connect("image", "out")` — wire public endpoint names.
- `graph.build()` — materialize the Graph into a `Run`.
- `run.push("image", [sample])` / `run.pull("out")` — named runtime I/O.

**When to use this**
- Custom orchestration where linear model calls are not enough (fan-out, fan-in, per-stream routing).
- Multistream scheduling (see chapter 014).
- Embedding model execution as one stage of a larger flow (see chapter 013).

**Prerequisites**
Chapter 003 (Graph basics).

**References**
- [Graph](/getting-started/programming-model/graph)
- [Public Graph](/getting-started/programming-model/graph)

## Learning Process
1. Build a minimal public Graph and push one deterministic tensor sample.
2. Use named endpoints rather than node IDs.
3. Pull graph output and validate stream/frame/timestamp metadata preservation.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/012_build_a_custom_data_graph/build_a_custom_data_graph.py
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_012_build_a_custom_data_graph
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_012_build_a_custom_data_graph
./build/tutorials-standalone/tutorial_012_build_a_custom_data_graph
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/012_build_a_custom_data_graph/build_a_custom_data_graph.cpp`
- Python: `tutorials/012_build_a_custom_data_graph/build_a_custom_data_graph.py`
