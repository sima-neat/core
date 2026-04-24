# 013 Embed a Model Inside a Graph

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Labels | graph, hybrid, stage-model, mpk |

## Concept

Drop a model into a graph as a single stage using `stage_model_executor`. This is the bridge pattern when you need graph-level orchestration (routing, scheduling) around model execution as one node in the flow.

Many production systems need graph-level control while still using model execution as a stage. The hybrid pattern — graph container + `StageModelExecutor` node — gives you both.

**APIs introduced**
- `pyneat.graph.nodes.StageModelExecutorOptions()` with model-backed fields.
- `pyneat.graph.nodes.stage_model_executor(opts, name)` — the graph-node form of a model stage.
- `graph.add(stage_node)` + `pyneat.graph.GraphSession(graph).build()` — same graph lifecycle as chapter 012.

**When to use this**
- Graph-level composition around model execution.
- Deterministic behavior in CI/dev even when model assets differ across environments.
- Safe migration path from simple stage graphs to model-backed hybrid graphs.

**Prerequisites**
Chapter 001 (Model). Chapter 012 (Graph basics).

**References**
- [Graph](/getting-started/programming-model/graph)
- [Model](/getting-started/programming-model/model)

## Learning Process
1. Prepare deterministic tensor samples that match model input contracts.
2. Build and run stage-model hybrid flow when MPK is available.
3. Fall back to stage-only graph flow when model path is unavailable.
4. Compare output kind/rank behavior and validate with `CHECK` + `SIGNATURE`.

## Run

Fetch the YOLOv8-s MPK once: `sima-cli modelzoo -v 2.0.0 get yolo_v8s`.

**Python:**
```bash
python3 share/sima-neat/tutorials/013_embed_model_inside_graph/embed_model_inside_graph.py \
  --mpk /path/to/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_013_embed_model_inside_graph \
  --mpk /path/to/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_013_embed_model_inside_graph
./build/tutorials-standalone/tutorial_013_embed_model_inside_graph \
  --mpk /path/to/yolo_v8s.tar.gz
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/013_embed_model_inside_graph/embed_model_inside_graph.cpp`
- Python: `tutorials/013_embed_model_inside_graph/embed_model_inside_graph.py`
