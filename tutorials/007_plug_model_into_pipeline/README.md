# 007 Plug a Model Into Your Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15 minutes |
| Model | yolo_v8s |
| Labels | graph, composition, patterns |

## Concept

Drop a model into a `Graph` with `model.graph()` and `model.graph(options)` — two composition patterns that differ in how much wiring control you take, so you know which fits a quick run versus a multi-camera deployment.

## Walkthrough

Chapter 003 built a graph node-by-node. That is the most explicit path, but once you have a `Model`, you rarely want to hand-wire its internals. `model.graph(...)` hands you the model's pipeline as a group you drop into a `Graph` with one `add(...)`. The interesting question is *how much* of the boundary that group brings with it — and that is what `ModelRouteOptions` controls.

This chapter contrasts two route configurations against the same model: a self-contained runnable graph that includes its own public input/output boundaries, and an attached graph that omits the input so it can hang off an upstream source (a camera, say) under an explicit name. By the end you will have composed both and printed each one's backend GStreamer string, so you can see exactly how the wiring differs.

### Compose a runnable model graph {#step-model-graph}

The first pattern asks the model for a fully runnable graph. Setting `include_input = true` and `include_output = true` on the route options tells `model.graph(opts)` to inject explicit public input and output boundaries around the model group, so the resulting `Graph` can be built and run on its own with nothing else attached. `graph.add(model.graph(opts))` is the whole composition — that single `add` is what every pattern reduces to underneath. Printing `describe_backend()` shows the generated GStreamer pipeline string.

**C++:** Route options are `Model::RouteOptions`; the graph is `simaai::neat::Graph`.

**Python:** Route options are `pyneat.ModelRouteOptions`; the graph is `pyneat.Graph`.

### Configure attach-time route options {#step-route-options}

The second pattern attaches the model under an upstream source rather than giving it its own input. Here `include_input = false` drops the public input boundary (the frames will come from elsewhere), `include_output = true` keeps the output, and `upstream_name`, `name_suffix`, and `buffer_name` make the wiring and element names explicit. Consistent naming like this is what keeps backend graphs readable and diagnosable in multi-camera or multi-model deployments.

### Attach the model group {#step-attached-graph}

With those options set, `graph.add(model.graph(opts))` injects the same model group, now wired to attach to the named upstream instead of carrying its own source. It is the identical `add` call as the first pattern — only the route options changed — which is the point: composition is one operation, and `ModelRouteOptions` is the dial that decides what boundary the group brings with it.

**C++:** Each variant prints its `describe_backend()` so you can compare the two backend strings; the file then also builds and runs a hand-wired direct `Input -> Output` graph to confirm the end-to-end path, printing `direct_rank=`.

**Python:** The attached variant prints `attached_graph_built=True` to confirm composition succeeded.

## Run

Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**.

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

Expected output (the C++ build prints each backend graph string, then the direct-graph rank):

```text
model_graph_backend=
...
attached_graph_backend=
...
direct_rank=3
[OK] 007_plug_model_into_pipeline
```

(The Python build prints `direct_graph_backend=` followed by the backend string, then `attached_graph_built=True`.) To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/007_plug_model_into_pipeline/plug_model_into_pipeline.cpp`
- Python: `tutorials/007_plug_model_into_pipeline/plug_model_into_pipeline.py`
