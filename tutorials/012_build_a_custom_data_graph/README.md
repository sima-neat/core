# 012 Build a Custom Data Graph

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Intermediate |
| Estimated Read Time | 15-20 minutes |
| Model | None |
| Labels | graph, traversal, metadata |

## Concept

Build the smallest useful public Neat `Graph` — one *named* `Input` wired to a *named* `Output` — then push a sample through it by name and verify the sample's metadata survives traversal.

## Walkthrough

Chapter 003 built an anonymous Input → Output graph and drove it with positional `run()` calls. Real orchestration — fan-out, fan-in, per-stream routing — needs to address endpoints *by name*, not by position. This chapter introduces that named-endpoint surface on the smallest possible graph, so you can see the naming and wiring mechanics in isolation before the multistream and embedded-model chapters build on them.

The public `Graph` is the application composition surface: you `add(...)` nodes, `connect(...)` named endpoints, `build()` once into a reusable `Run`, then `push("image", ...)` and `pull("out", ...)` by name. By the end you will have pushed one tensor `Sample` through a named graph and confirmed its `stream_id`, `frame_id`, and `pts_ns` came out unchanged — proof the runtime preserves metadata end to end.

### Compose the graph {#step-compose-graph}

Add two nodes. `Input("image")` declares a push endpoint named `image`; `Output("out")` declares a pull endpoint named `out`. The names are the contract — they are exactly the strings you will pass to `push(...)` and `pull(...)` later. Naming endpoints (rather than relying on add-order) is what makes larger graphs with multiple inputs or outputs unambiguous to drive.

**C++:** Nodes come from `simaai::neat::nodes::Input("image")` and `nodes::Output("out")`.

**Python:** Nodes come from `pyneat.nodes.input("image")` and `pyneat.nodes.output("out")`.

### Wire the endpoints {#step-connect-endpoints}

`connect("image", "out")` declares the edge: frames pushed to `image` flow to `out`. With only two nodes this is the entire topology, but `connect(...)` is the same call you would use to build branches and merges in a larger graph. We then print `graph.describe()` to dump the composed topology — a quick sanity check that the graph is wired the way you intended before building.

### Build and push a sample {#step-build-and-push}

`build()` (with no priming sample needed here) materializes the description into a runnable `Run`. We then construct one deterministic tensor `Sample` — an 8×8×3 RGB image carrying a known `stream_id`, `frame_id`, and `pts_ns` — and `push(...)` it to the `image` endpoint by name. The sample's metadata is what we will check on the other side.

**C++:** `push(...)` returns a bool; on failure we surface `run.last_error()`. The sample is built by `make_sample()`.

**Python:** `push("image", [sample])` takes a list of samples. The sample is built by `make_rgb_sample()`.

### Pull the output and verify metadata {#step-pull-and-verify}

`pull("out", ...)` retrieves the result from the named output endpoint with a timeout, after which we `close()` the run. Because there is no transform between input and output, a correct pipeline returns the same logical sample — so reading back `stream_id`, `frame_id`, and `pts_ns` and seeing the values we pushed confirms the runtime preserved per-sample metadata through traversal. That guarantee is what lets downstream stages trust frame identity and timestamps.

## Run

Run it and you should see the graph description followed by the round-tripped metadata. Run the **Python** and **C++ (prebuilt)** commands from the **Neat install root** (the directory that contains `share/` and `lib/`); run the **build from source** commands from the **repo root**. This chapter needs no model archive.

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

Expected output (preceded by the `graph.describe()` dump):

```text
stream=graph frame=42 pts_ns=123456789
[OK] 012_build_a_custom_data_graph
```

(The Python build prints `stream_id=graph frame_id=42 pts_ns=123456789`.) To integrate this chapter's C++ source into your own project with a custom `CMakeLists.txt` (no extras folder required), see [How to Run Tutorials](/tutorials#compile-a-copy-yourself) on the landing page.

## Source Files
- C++: `tutorials/012_build_a_custom_data_graph/build_a_custom_data_graph.cpp`
- Python: `tutorials/012_build_a_custom_data_graph/build_a_custom_data_graph.py`
