# 007 Plug a Model Into Your Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | graph, composition, patterns |

## Concept

Three ways to drop a model into a `Graph` — direct node composition, `model.graph()`, and `model.graph(options)` — so you know which pattern fits which context. All three produce a runnable Graph; they differ in explicitness and control.

The three composition patterns shown:
- **Direct Graph**: add `Input` + `Output` nodes yourself with `graph.add(...)`. Most explicit; useful when you need full control over the wiring.
- **Model-default Graph**: `model.graph()` injects the model's default pipeline group with sensible defaults. Shortest path for most cases.
- **Model-attached Graph**: `model.graph(ModelRouteOptions)` controls appsrc/appsink inclusion and stage naming when attaching model groups into larger pipelines.

Why this matters:
- Teams often start with direct Graphs for clarity, then move to model-backed composition as systems scale.
- `ModelRouteOptions` keeps graph wiring explicit in multi-camera or multi-model deployments.
- Consistent naming (`upstream_name`, `name_suffix`, `buffer_name`) improves diagnostics and backend graph readability.

**APIs introduced**
- `model.graph()` — inject the default model pipeline.
- `pyneat.ModelRouteOptions()` + `model.graph(opts)` — attach with explicit options.
- `graph.add(group_or_node)` — what all three patterns reduce to underneath.

**Prerequisites**
Chapter 001 (Model). Chapter 002 or 003 (Graph basics).

**References**
- [Graph](/getting-started/programming-model/graph)
- [Model](/getting-started/programming-model/model)

## Learning Process
1. Build a minimal direct Graph and validate it can run end-to-end.
2. Construct model-backed Graph variants (`model.graph()` and `model.graph(options)`).
3. Compare generated backend graphs with `--print-gst` to understand composition differences.

## Run

**Python:**
```bash
python3 share/sima-neat/tutorials/007_plug_model_into_pipeline/plug_model_into_pipeline.py \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (prebuilt):**
```bash
./lib/sima-neat/tutorials/tutorial_007_plug_model_into_pipeline \
  --model /tmp/yolo_v8s.tar.gz
```

**C++ (build from source):**
```bash
./build.sh --target tutorial_007_plug_model_into_pipeline
./build/tutorials-standalone/tutorial_007_plug_model_into_pipeline \
  --model /tmp/yolo_v8s.tar.gz
```

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/007_plug_model_into_pipeline/plug_model_into_pipeline.cpp`
- Python: `tutorials/007_plug_model_into_pipeline/plug_model_into_pipeline.py`
