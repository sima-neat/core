---
title: Public Graph and graph helpers
description: How to use the public `Graph`, plus how it relates to builder and runtime graph helpers.
sidebar_position: 3
---

# Public Graph and graph helpers

Most application code should use the public `simaai::neat::Graph`. Lower-level runtime graph
helpers exist for compiler/runtime internals and focused tests, but they are not the normal
user-facing composition API.

## `simaai::neat::Graph` - public pipeline graph

`Graph` is the evolution of the old `Graph` composition surface:

- `add(...)` appends to the default linear chain.
- `connect(...)` creates explicit graph topology between fragments.
- `build(...)` returns one public `Run`.

Single-input / single-output graphs keep the compact API:

```cpp
sima::Graph g;
g.add(sima::nodes::Input());
g.add(model);
g.add(sima::nodes::Output());

auto run = g.build();
run.push(sima::TensorList{image});
auto out = run.pull();
```

For multi-input / multi-output graphs, name public endpoints with `Input("name")` and
`Output("name")`. `Graph("name")` is only a label for diagnostics/visualization; it does not
declare an endpoint.

If the endpoint names come from a model, ask the model graph what they are instead of guessing:

```cpp
sima::Graph model_route = model.graph();
auto model_inputs = model_route.inputs();    // e.g. {"image_l", "image_uv"}
auto model_outputs = model_route.outputs();  // e.g. {"classes"} or one aggregate model output

sima::Graph left;
left.add(sima::nodes::Input("image_l"));

sima::Graph right;
right.add(sima::nodes::Input("image_uv"));

sima::Graph classes;
classes.add(sima::nodes::Output("classes"));

sima::Graph app;
app.connect(left, model);
app.connect(right, model);
app.connect(model, classes);

auto run = app.build();
run.push("image_l", sima::TensorList{left_tensor});
run.push("image_uv", sima::TensorList{right_tensor});
auto y = run.pull("classes");
```

Model inputs are matched by exact logical endpoint name. For example, `Input("a_new_name_image_l")`
does **not** bind to model input `"image_l"` implicitly. Rename/adapter graphs must be explicit.

For an output collection with multiple unnamed `Output()` nodes, a Graph label may be used as the
generated-name prefix. This is still not endpoint matching; it only chooses stable names for the
outputs that will be exposed by the built `Run`:

```cpp
sima::Graph classes("classes");
classes.add(sima::nodes::Output());
classes.add(sima::nodes::Output());
classes.add(sima::nodes::Output());

// Pull as "classes_0", "classes_1", and "classes_2".
```

Calling plain `push(...)` or `pull()` on a graph with multiple possible inputs/outputs fails closed
with a diagnostic listing the available endpoint names.

## Saving a built graph run for visualization

`Graph::build()` produces a `Run`. A `Run` knows both:

- the user-facing graph shape: named inputs, named outputs, and connected fragments; and
- the lowered runtime shape: pipeline segments, runtime edges, counters, latency, and optional
  board-power telemetry.

You can save that information as JSON in two ways.

### 1. Build-time snapshot

Use this when CI or a debugging tool needs a graph artifact even before samples have run:

```cpp
sima::RunOptions opt;
opt.graph_run_export.path = "/tmp/startup.graph_run.json";
opt.graph_run_export.label = "resnet_startup";

sima::Run run = app.build(opt);
```

This file is an **initial topology snapshot**. It is useful for answering "what graph did build
produce?" but throughput and latency counters may still be zero because no frames have been pushed or
pulled yet.

Python:

```python
opt = pyneat.RunOptions()
opt.graph_run_export.path = "/tmp/startup.graph_run.json"
opt.graph_run_export.label = "resnet_startup"

run = app.build(opt)
```

### 2. Post-run snapshot with metrics

Use this after the workload has run/drained when you want the graph plus runtime measurements:

```cpp
sima::Run run = app.build();
run.push("image", sima::TensorList{image});
auto y = run.pull("classes");

sima::GraphRunExportOptions export_opt;
export_opt.label = "resnet_with_metrics";
export_opt.metadata = {{"test_name", "smoke"}};

std::string err;
if (!sima::save_graph_run_json(run, "/tmp/final.graph_run.json", export_opt, &err)) {
  throw std::runtime_error(err);
}
```

Python:

```python
run = app.build()
run.push("image", [image])
y = run.pull("classes")

export_opt = pyneat.GraphRunExportOptions()
export_opt.label = "resnet_with_metrics"
export_opt.metadata = [("test_name", "smoke")]

run.save_graph_run_json("/tmp/final.graph_run.json", export_opt)
```

The exporter is a snapshot operation: it does **not** stop the run. If you want final numbers for a
finite stream, close or drain the run first, then save. If board power monitoring was enabled in
`RunOptions`, the power summary is included with the same JSON.

The JSON contract is versioned as `sima.neat.graph_run` schema version `1`. The checked-in schema
artifact lives at `schemas/graph_run_v1.schema.json`, and the dependency-free validator used by CI
lives at `tests/perf/tools/graph_run_schema.py`.

The export intentionally carries both **what the user authored** and **what NEAT executed**:

- `graph.public_view` is the public endpoint graph: named `Input`/`Output` boundaries and the
  `connect()` edges the application wrote.
- `graph.lowered_view` is the runtime graph NEAT built internally: pipeline segments, inserted
  branch/combine stages, and runtime edges.
- `run.identity` records a per-run UUID, creation/close timestamps when known, hostname, PID, and
  command line. This is the "what did I actually run?" block for CI artifacts and bug reports.
- Nodes may include `source`, `sink`, and `model` blocks. For example, app-pushed inputs are marked
  as `source.kind = "app_push"`, pulled outputs as `sink.kind = "appsink"`, file/RTSP/UDP nodes carry
  URI-style locations when those are known, and model-bound stages carry model lineage/source-path
  metadata. These fields are best-effort and schema-extensible: absence means the node type did not
  expose that information yet, not that the graph is invalid.

To render the file without internet access or CDN assets:

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json -o /tmp/final.graph_run.html
```

The generated HTML is self-contained and can show either the public endpoint graph or the lowered
runtime graph:

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view public
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view lowered
```


## Named boundaries and materialization

Named `Input` and `Output` nodes are the public contract of a Graph fragment.
They are deliberately higher level than the runtime objects used to move buffers.

In other words:

```cpp
Graph route("route");
route.add(nodes::Input("image"));
route.add(model);
route.add(nodes::Output("classes"));
```

means:

```text
route(image) -> classes
```

It does **not** always mean "allocate a separate physical runtime input and a separate physical
runtime output here". Whether a boundary becomes physical runtime I/O depends on where the fragment
is used in the final app graph.

### Boundary materialization rules

Before executable pipeline construction, public `Graph::build()` normalizes boundaries:

| Boundary declaration | Materialized when... | Elided when... |
|---|---|---|
| `nodes::Input("name")` | no upstream graph is connected to it; it must be a public `Run::push("name", ...)` endpoint | an upstream graph feeds it; it is only an internal fragment parameter |
| `nodes::Output("name")` | no downstream graph consumes it; it must be a public `Run::pull("name")` endpoint | a downstream graph consumes it; it is only an internal fragment return value |

"Elided" does not mean the user's graph loses information. It means the executable pipeline does
not allocate a useless source/sink for that declaration. The compiler keeps a provenance map so
`describe()`, validation errors, runtime diagnostics, metrics, and graph visualization can still
refer back to the user-facing endpoint name.

### Why this exists

This is a correctness rule, not just an optimization.

Without boundary materialization, composing reusable fragments would create hidden runtime I/O in the
middle of the application. For example, this innocent app:

```cpp
Graph app;
app.connect(camera, route);
app.connect(route, display);
```

could accidentally lower to:

```text
camera -> route.Input -> model -> route.Output
                                -> display.Output
```

That shape has two output sinks in what the user intended to be one path. It can fail validation,
create backpressure, or force avoidable memory handoffs. The high-quality behavior is to treat
`route.Input` and `route.Output` as internal declarations and lower the executable shape to:

```text
camera -> model -> display
```

while still remembering that the logical route was `camera -> route("image") -> route("classes") -> display`.

### Relationship to the lower-level runtime graph

The lower-level `simaai::neat::graph::Graph` still has runtime ports because the scheduler needs
precise internal routing. Public `simaai::neat::Graph` intentionally does not expose those runtime
ports for normal composition. Public composition is expressed with Graph fragments and named
`Input`/`Output` declarations.

When a public graph is built, the compiler translates public endpoint names into the internal port
and queue layout. That translation is private and deterministic. Customers should reason in terms of
ML-style named inputs and outputs, not runtime port strings.

### Multiple producers or consumers

A named endpoint must have a clear meaning. If a public output has multiple producers, the graph must
say how those producers should be combined. Use an explicit helper such as:

```cpp
auto pair = graphs::Combine({"left", "right"}, "pair", CombinePolicy::ByFrame);
```

A named output used for several consumers is a branch and should be expressed with:

```cpp
auto branch = graphs::Branch("image", {"preview", "model"});
```

This keeps synchronization, backpressure, and drop behavior visible in code instead of hiding it in
an accidental topology.

## Legacy builder topology helper removed

Older NEAT versions had a separate outer-namespace `BuilderGraph` helper with node IDs and
`connect(id, id)`. That concept is intentionally removed from the public graph model. Reusable
fragments are now plain `Graph`s:

```cpp
sima::Graph source("image");
source.add(sima::nodes::groups::RtspDecodedInput(opts));

sima::Graph route;
route.add(model);

sima::Graph app;
app.connect(source, route);
```

`Graph::add(fragment)` splices a linear fragment; `Graph::connect(...)` creates explicit topology.
There is no second public builder DAG to learn.

## Saving and loading connected Graphs

`Graph::save(path)` writes the **public graph composition**, not just the lowered GStreamer string.
For connected graphs the file contains:

- the public nodes and their endpoint names;
- explicit endpoint edges and their public endpoint labels;
- `OutputOptions`, including `CombinePolicy`;
- fragment provenance for model routes.

Model-route provenance is important. A `Model` fragment is not just a list of GStreamer snippets:
it also carries input/output endpoint names derived from the model archive, route options, and
input-route processor metadata for multi-input models. When a saved graph contains a model fragment,
NEAT stores the model archive path plus the model construction options and route options needed to
rehydrate that fragment on `Graph::load(path)`. If the original model archive is no longer present,
loading fails with an actionable error instead of silently building a stale or incomplete route.

## `simaai::neat::graph::Graph` - internal runtime graph

Defined in [`graph/Graph.h`](/reference/cppapi/files/include-graph-graph-h). This is the
lower-level actor-style runtime graph used by NEAT internals. It schedules `StageExecutor` nodes via
mailboxes and uses named runtime ports on each node so compiler-generated stages can route data
precisely.

Public application code should normally not author this graph directly. Public `simaai::neat::Graph`
lowers to this runtime substrate when it needs scheduler stages, branch/combined runtime nodes, or
other non-linear execution machinery.

In Python, the old `pyneat.graph` module is a deprecated compatibility alias for the internal
runtime substrate. New application code should use `pyneat.Graph` plus `pyneat.graphs.branch(...)`
and `pyneat.graphs.combine(...)`. Internal runtime tests that intentionally exercise the lowered
port graph can use `pyneat._graph`.

## Helper comparison

| Aspect | `simaai::neat::Graph` (public) | `simaai::neat::graph::Graph` (runtime/internal) |
|--------|----------------------------------|-----------------------------------------------|
| Purpose | User-facing pipeline/model composition | Lowered runtime substrate for scheduler/stage internals |
| Threads | Owns runtime after `build()` | Owns scheduler threads, mailboxes, queues |
| Backend | Compiles to the unified `Run` backend | Internal graph runtime |
| Ports | `connect(...)`; named public endpoints through `Run` | Edges carry named runtime `from_port` / `to_port` |
| Common case | `g.add(model); run.push(...); run.pull(...)` | Internal compiler/runtime tests |

## How to tell them apart in code

- `simaai::neat::Graph` lives in the outer `simaai::neat` namespace and is the public API.
- `simaai::neat::graph::Graph` lives in the inner `simaai::neat::graph` namespace and is a
  lower-level runtime/compiler type. Its `connect()` takes named runtime port strings; nodes hold
  `std::shared_ptr<simaai::neat::graph::Node>`.

## Further reading

- "Graphs" - sections 0.14, 10, and 73 of the design deep dive ([Architecture](/contribute/architecture)).
- [`graph/Graph.h`](/reference/cppapi/files/include-graph-graph-h)
- [`graph/GraphBuild.h`](/reference/cppapi/files/include-graph-graphbuild-h)
- [`graph/StageExecutor.h`](/reference/cppapi/files/include-graph-stageexecutor-h)
