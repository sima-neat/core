# 013 Embed a Model Inside a Graph

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | yolo_v8s |
| Labels | graph, hybrid, model, mpk |

## Concept

Drop a model into a public `Graph` with `graph.add(model)`. This is the bridge pattern when you need graph-level orchestration (routing, scheduling, extra inputs/outputs) around model execution without reaching into the internal runtime graph.

Many production systems need graph-level control while still using model execution as a reusable fragment. The public pattern is now:

```python
model = pyneat.Model("/models/yolo_v8s.tar.gz")

graph = pyneat.Graph()
graph.add(pyneat.nodes.input("image"))
graph.add(model)
graph.add(pyneat.nodes.output("result"))
```

NEAT lowers the model fragment into the correct internal runtime execution plan at build time. Application code does not use `StageModelExecutorOptions`, node IDs, or `pyneat.graph`.

**APIs introduced**
- `pyneat.Model(path)` — load a compiled model archive.
- `pyneat.Graph()` — public composition surface.
- `graph.add(model)` — append the model route as a reusable Graph fragment.
- `pyneat.nodes.input(name)` / `pyneat.nodes.output(name)` — public named boundaries.

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
1. Load a model archive.
2. Add the model directly to a public Graph.
3. Inspect the composed Graph without using low-level runtime graph node IDs.
4. Use later tutorials for full model execution with real inputs and accuracy checks.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/013_embed_model_inside_graph/embed_model_inside_graph.py \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_013_embed_model_inside_graph \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_013_embed_model_inside_graph
./build/tutorials-standalone/tutorial_013_embed_model_inside_graph \
  --model /tmp/yolo_v8s.tar.gz
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/013_embed_model_inside_graph/embed_model_inside_graph.cpp`
- Python: `tutorials/013_embed_model_inside_graph/embed_model_inside_graph.py`
