# 013 Embed a Model Inside a Graph

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Advanced |
| Estimated Read Time | 20-25 minutes |
| Model | yolo_v8s |
| Labels | graph, hybrid, model, mpk |

## Concept

Drop a compiled model into a public `Graph` with `graph.add(model)`, so you get graph-level orchestration (routing, scheduling, extra inputs/outputs) wrapped around model execution without reaching into the internal runtime graph.

## Walkthrough

Chapter 003 built a graph out of bare input/output nodes; chapter 001 ran a model as a standalone object. This chapter joins the two: a `Model` is itself a graph-compatible node, so you can compose it into a public `Graph` exactly like any other stage. That is the bridge pattern production systems reach for when they need graph-level control — multiple inputs, named outputs, custom routing — while still treating model execution as one reusable fragment.

The key idea is that you never touch the low-level runtime graph, `StageModelExecutorOptions`, or internal node IDs. You hand the model to `graph.add(...)`, and NEAT lowers that fragment (preprocess / inference / postprocess as needed) into the correct internal execution plan at build time. By the end you will have composed a model into a public graph, printed the composed topology, and read back the model's output cardinality.

### Load the model {#step-load-model}

Construction loads the compiled archive and prepares it for execution, just as in chapter 001. Here we take only the path — no options object — because this chapter is about composition, not preprocessing. The resulting `Model` is now an object the graph layer understands.

### Compose the model into a graph {#step-compose-graph}

This is the whole point of the chapter. A fresh `Graph` gets three nodes added in order: a named input boundary, the model itself, and a named output boundary. Because `Model` is graph-compatible, `add(model)` appends the entire model route as a single fragment — there is no special API and no reaching into the runtime. Printing `graph.describe()` shows the composed topology so you can confirm the model slotted in between the named boundaries.

**C++:** Boundaries come from `simaai::neat::nodes::Input("image")` and `nodes::Output("result")`; the model is passed directly to `graph.add(model)`.

**Python:** Boundaries come from `pyneat.nodes.input("image")` and `pyneat.nodes.output("result")`; the model is passed directly to `graph.add(model)`.

### Inspect the model {#step-inspect-model}

Finally we read back what the model fragment actually contributes. This confirms the model loaded correctly and lets you see the output topology the graph will produce downstream.

**C++:** `model.info()` returns an info struct; we print `model_name` plus `output_topology.physical_outputs` and `logical_outputs` so the wiring of the model's outputs is explicit.

**Python:** The binding for this chapter simply prints a confirmation line that the model fragment was added to the public graph.

## Run

This chapter needs a model archive (`yolo_v8s`). Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**.

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

Expected output (the C++ build also prints the composed graph description first):

```text
model=yolo_v8s physical_outputs=1 logical_outputs=1
[OK] 013_embed_model_inside_graph
```

(The Python build prints the graph description followed by `model fragment added to public Graph`.)

To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/013_embed_model_inside_graph/embed_model_inside_graph.cpp`
- Python: `tutorials/013_embed_model_inside_graph/embed_model_inside_graph.py`
