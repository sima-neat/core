---
title: Building applications with Graph
description: How to compose Models, Nodes, named inputs/outputs, branches, combines, and Runs with the public Graph API.
sidebar_position: 1
slug: /develop-apps/advanced-concepts/graphs
---

# Building applications with Graph

Use `Model` when you only want to load and run one compiled model archive. Use `Graph` when you
want to build an application around models and nodes: add public inputs and outputs, connect reusable
fragments, branch streams, combine streams, validate the app, and save or visualize what actually
ran.

The mental model is intentionally small:

| Concept | Meaning |
|---|---|
| `Model` | A compiled model archive loaded from disk, for example `resnet50.tar.gz` or `yolov8.tar.gz`. |
| `Node` | One processing step: an input, output, transform, source, sink, model stage, or helper stage. |
| `Graph` | The application wiring plan: what nodes/fragments exist and how data flows between them. |
| `Run` | The live execution handle returned by `Graph::build()`: push inputs, pull outputs, collect metrics, stop. |

In short:

```text
Graph = what to run
Run   = the running instance
```

Most application code should use the public `simaai::neat::Graph`. Do not build applications with
lower-level `simaai::neat::graph::*` runtime/compiler helpers; those are internal implementation
substrate and focused-test utilities.

## When do I need a Graph?

| Goal | Recommended API |
|---|---|
| Run one model on one input | `Model::run(...)` or `Model::build(...)` |
| Add application input/output boundaries around a model | `Graph` |
| Compose a model with custom processing nodes | `Graph::add(...)` |
| Reuse a sub-pipeline in multiple apps | Return/pass a `Graph` fragment |
| Route multiple inputs or outputs | Named `nodes::Input(...)` / `nodes::Output(...)` plus `connect(...)` |
| Branch one stream to several consumers | `graphs::Branch(...)` |
| Combine several streams into one logical output | `graphs::Combine(...)` with a `CombinePolicy` |
| Save or visualize the executed topology and metrics | `save_run_json(run, ...)` |

## First Graph: one input, one model, one output

This is the smallest complete app-style graph:

```cpp
#include <neat.h>

#include <iostream>

namespace neat = simaai::neat;

int main() {
  neat::Model model("resnet50.tar.gz");

  neat::Graph app;
  app.add(neat::nodes::Input("image"));
  app.add(model);
  app.add(neat::nodes::Output("classes"));

  neat::Run run = app.build();

  neat::Tensor image = /* create or load an image tensor */;
  run.push("image", neat::TensorList{image});

  std::optional<neat::Sample> result = run.pull("classes", /*timeout_ms=*/1000);
  if (result) {
    // Consume result->tensors, result->detections, or other Sample metadata.
  }

  run.stop();
}
```

Line by line:

- `nodes::Input("image")` declares a public input door named `image`.
- `app.add(model)` inserts the model's selected route into the graph.
- `nodes::Output("classes")` declares a public output door named `classes`.
- `app.build()` validates and compiles the entire graph and returns a `Run`.
- `run.push("image", ...)` sends data into the named input.
- `run.pull("classes", ...)` receives data from the named output.

The same shape from Python:

```python
import pyneat

model = pyneat.Model("resnet50.tar.gz")

app = pyneat.Graph()
app.add(pyneat.nodes.input("image"))
app.add(model)
app.add(pyneat.nodes.output("classes"))

run = app.build()

image = ...  # Create or load a tensor-compatible object.
run.push("image", [image])

result = run.pull("classes", timeout_ms=1000)
run.stop()
```

Python `Run.push(...)` expects a batch-like sequence. Pass `[tensor]` or `[sample]`, not a bare
single tensor/sample object.

## Running a Graph

A built `Run` accepts the same public payload types used elsewhere in NEAT:

| Payload | Use when |
|---|---|
| `TensorList` | You are passing tensors and do not need extra sample metadata. |
| `Sample` | You need timestamps, `frame_id`, `stream_id`, text/audio/video metadata, detections, or EOS. |
| `std::vector<cv::Mat>` | You want image convenience input from OpenCV. |

Common C++ calls:

```cpp
run.push(neat::TensorList{image});
run.push("image", neat::TensorList{image});

run.push(sample);
run.push("image", sample);

auto out = run.pull(/*timeout_ms=*/1000);
auto named = run.pull("classes", /*timeout_ms=*/1000);

neat::TensorList tensors = run.pull_tensors("classes", 1000);
neat::Sample sample_out = run.pull_samples("classes", 1000);
```

Use `pull(...)` when timeout/closed should return an empty `std::optional`. Use `pull_tensors(...)`
or `pull_samples(...)` when you want a typed convenience helper that throws on timeout/error.

For finite app-pushed streams, close input and drain before collecting final metrics:

```cpp
run.close_input();
while (auto out = run.pull("classes", 1000)) {
  // Drain remaining output.
}
run.stop();
```

## `build()` versus `build(first_input)`

Most graphs can be built without an input sample:

```cpp
neat::Run run = app.build();
```

Use this when the graph already declares enough shape/caps information, or when the graph owns its
source nodes, such as RTSP/file/still-image inputs.

Seeded build gives NEAT the first input during build:

```cpp
neat::RunOptions opt;
opt.startup_preflight = true;

neat::Run run = app.build(neat::TensorList{first_image}, neat::RunMode::Async, opt);
```

Use this when the first input should seed shape/format adaptation before streaming starts. With
`startup_preflight = true` (the default for seeded build paths), NEAT may push/pull the seed once to
catch first-sample failures during build instead of returning a `Run` that immediately fails later.

For throughput, latency, and power numbers, save metrics after the actual workload has run, not
immediately after build.

## Graph names are not endpoint names

:::warning
`Graph("name")` is a label for diagnostics, saved graph files, and visualization. It does **not**
declare a public input or output named `name`.
:::

Wrong mental model:

```cpp
neat::Graph camera("image");
// This does not make run.push("image", ...) valid by itself.
```

Correct endpoint declaration:

```cpp
neat::Graph camera("camera_route");
camera.add(neat::nodes::Input("image"));
```

And for an output:

```cpp
neat::Graph classifier("classifier");
classifier.add(neat::nodes::Output("classes"));
```

Think of `Input("image")` and `Output("classes")` as the public doors of a graph fragment. The graph
name is just the sign on the building.

## Inspect endpoint names instead of guessing

Before build, inspect the logical public endpoints declared by a graph:

```cpp
for (const auto& name : app.inputs()) {
  std::cout << "graph input: " << name << "\n";
}
for (const auto& name : app.outputs()) {
  std::cout << "graph output: " << name << "\n";
}
```

After build, inspect what the `Run` actually accepts:

```cpp
for (const auto& name : run.input_names()) {
  std::cout << "run input: " << name << "\n";
}
for (const auto& name : run.output_names()) {
  std::cout << "run output: " << name << "\n";
}
```

Use this for model routes and any multi-input/multi-output app. Endpoint matching is exact:
`Input("image_l")` can bind to a model input named `image_l`; `Input("my_random_name")` does not.

## Unnamed convenience APIs

For one-input / one-output graphs, you may omit endpoint names:

```cpp
neat::Graph app;
app.add(neat::nodes::Input());
app.add(model);
app.add(neat::nodes::Output());

neat::Run run = app.build();
run.push(neat::TensorList{image});
auto result = run.pull(1000);
```

This is convenient for quick scripts and tests. For nontrivial applications, prefer names.

If a graph has multiple possible inputs or outputs, unnamed `push(...)` or `pull()` fails closed and
reports the available names. That failure is intentional: NEAT should not guess which camera, tensor,
or output head you meant.

## Models are Graph fragments

A `Model` can be added directly to a graph:

```cpp
neat::Model yolo("yolov8.tar.gz");

neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

`Graph::add(model)` inserts the model route selected from the archive and model options. That route
may include preprocess, MLA inference, postprocess, tensor conversion, and detection decode stages.
You do not have to manually call `model.graph()` for the common linear case.

For advanced composition, inspect or reuse the route as a `Graph` fragment:

```cpp
neat::Graph route = yolo.graph();

auto model_inputs = route.inputs();
auto model_outputs = route.outputs();
```

### Multi-input models

For multi-input models, do not guess names. Ask the route:

```cpp
neat::Graph route = model.graph();

for (const auto& name : route.inputs()) {
  std::cout << "model expects input: " << name << "\n";
}
```

Then name your upstream fragments to match the model's input names:

```cpp
neat::Graph left_camera;
left_camera.add(neat::nodes::Input("image_l"));

neat::Graph uv_camera;
uv_camera.add(neat::nodes::Input("image_uv"));

neat::Graph app;
app.connect(left_camera, route);  // Binds image_l -> model image_l.
app.connect(uv_camera, route);    // Binds image_uv -> model image_uv.
```

If `left_camera` declared `Input("a_new_name_image_l")`, it would not bind to `image_l`. Add a small
adapter graph with the correct endpoint name instead of relying on implicit renaming.

### Standalone model Graphs

By default, `model.graph()` returns a reusable model fragment with open named endpoints. If you want
the returned graph to be runnable by itself, request explicit public input/output nodes:

```cpp
neat::Model::RouteOptions route_opt;
route_opt.include_input = true;
route_opt.include_output = true;

neat::Graph standalone = model.graph(route_opt);
neat::Run run = standalone.build();
```

For advanced/debug use, a model route can expose individual physical outputs:

```cpp
route_opt.expose_all_outputs = true;
```

Leave this disabled unless you specifically need separate physical output buffers. The default model
behavior is to expose the logical model output expected by the route contract. If the model has only
one physical output, `expose_all_outputs = true` still exposes only one output.

## `add()` versus `connect()`

There are two composition tools:

| API | Meaning | Use when |
|---|---|---|
| `add(x)` | Append or splice into the current linear chain. | You mean “next step in the same pipeline.” |
| `connect(a, b)` | Wire two graph fragments by named endpoints. | You are composing reusable fragments or building topology. |
| `connect("a", "b")` | Wire two endpoints already declared inside the same graph. | You are building a small helper fragment. |

Linear composition:

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::Output("classes"));
```

Fragment composition:

```cpp
neat::Graph app;
app.connect(camera, model_route);
app.connect(model_route, output_sink);
```

Internal endpoint wiring inside a helper fragment:

```cpp
neat::Graph pass_through("pass_through");
pass_through.add(neat::nodes::Input("in"));
pass_through.add(neat::nodes::Output("out"));
pass_through.connect("in", "out");
```

The key rule: `add()` means a linear chain. `connect()` means graph topology.

## Reusable Graph fragments

Functions can return reusable graph fragments:

```cpp
neat::Graph make_classifier(neat::Model& model) {
  neat::Graph g("classifier");
  g.add(neat::nodes::Input("image"));
  g.add(model);
  g.add(neat::nodes::Output("classes"));
  return g;
}
```

Use a reusable fragment linearly:

```cpp
neat::Graph classifier = make_classifier(model);

neat::Graph app;
app.add(classifier);
```

Or wire fragments explicitly:

```cpp
neat::Graph app;
app.connect(camera, classifier);
app.connect(classifier, class_sink);
```

If `add()` after a branch would be ambiguous, NEAT fails and tells you to use `connect(...)` instead.
That is better than silently appending to the wrong branch.

## Branching one stream

Use `graphs::Branch` when one input stream should go to multiple named outputs:

```cpp
neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_input"});
```

Meaning:

```text
image -> preview
      -> model_input
```

Example:

```cpp
neat::Graph camera;
camera.add(neat::nodes::Input("image"));

neat::Graph preview;
preview.add(neat::nodes::Output("preview"));

neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_input"});

neat::Graph app;
app.connect(camera, branch);
app.connect(branch, preview);
```

When connecting a branch to a model, choose the branch output name to match the model input name:

```cpp
neat::Graph route = model.graph();
for (const auto& name : route.inputs()) {
  std::cout << "choose a branch output matching: " << name << "\n";
}
```

Branching is explicit because it affects queues and backpressure. If one branch is slow, it can slow
or drop relative to another branch depending on the output options and downstream graph.

Python:

```python
branch = pyneat.graphs.branch("image", ["preview", "model_input"])
```

## Combining multiple streams

Use `graphs::Combine` when several input streams should become one logical output:

```cpp
neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "stereo",
                                         neat::CombinePolicy::ByFrame);
```

Meaning:

```text
left  --\
        +--> stereo
right --/
```

Policies:

| Policy | Meaning |
|---|---|
| `CombinePolicy::None` | Do not combine automatically. Multiple producers to one output fail closed. |
| `CombinePolicy::ByFrame` | Match samples with exactly the same `Sample::frame_id`. Missing frame IDs fail; there is no PTS fallback. |
| `CombinePolicy::ByPts` | Match samples with exactly the same `Sample::pts_ns` presentation timestamp. Missing PTS fails; there is no frame-id fallback. |

Plain language:

- `ByFrame` means “give me left and right samples with the same frame number.”
- `ByPts` means “give me samples with the same media timestamp.”

Example:

```cpp
neat::Graph left;
left.add(neat::nodes::Input("left"));

neat::Graph right;
right.add(neat::nodes::Input("right"));

neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "stereo",
                                         neat::CombinePolicy::ByFrame);

neat::Graph app;
app.connect(left, pair);
app.connect(right, pair);

neat::Run run = app.build();
run.push("left", left_sample_with_frame_id_42);
run.push("right", right_sample_with_frame_id_42);
auto stereo = run.pull("stereo", 1000);
```

Python:

```python
pair = pyneat.graphs.combine(["left", "right"], "stereo", pyneat.CombinePolicy.ByFrame)
```

If samples do not carry the required key, the combine stage fails with a diagnostic instead of
guessing.

## Sources and sinks

There are two ways data enters a graph and two ways it leaves.

### App-pushed input

Use `nodes::Input(...)` when application code will push data:

```cpp
app.add(neat::nodes::Input("image"));
run.push("image", neat::TensorList{image});
```

### Graph-owned input source

Use a source node or source fragment when the graph owns the data source:

```cpp
app.add(neat::nodes::RTSPInput("rtsp://camera/stream"));
```

or a reusable decoded RTSP fragment:

```cpp
neat::nodes::groups::RtspDecodedInputOptions opt;
opt.url = "rtsp://camera/stream";

app.add(neat::nodes::groups::RtspDecodedInput(opt));
```

When a graph owns its source, you usually call `build()` and then pull outputs; you do not push into
that source from application code.

### App-pulled output

Use `nodes::Output(...)` when application code should pull results:

```cpp
app.add(neat::nodes::Output("detections"));
auto out = run.pull("detections", 1000);
```

### Graph-owned output sink

Use an output sink node or group when the graph should write results itself:

```cpp
neat::UdpOutputOptions udp;
udp.host = "192.0.2.10";
udp.port = 5000;

app.add(neat::nodes::UdpOutput(udp));
```

Server-style RTSP output is also available for graphs that are built for that mode:

```cpp
neat::RtspServerHandle server = app.run_rtsp(rtsp_options);
```

## Validation and diagnostics

Validate before building when you want a structured report without starting runtime resources:

```cpp
neat::GraphReport report = app.validate();
if (!report.error_code.empty()) {
  std::cerr << report.repro_note << "\n";
}
```

Catch `NeatError` around build/run/push/pull calls:

```cpp
try {
  neat::Run run = app.build();
} catch (const neat::NeatError& e) {
  std::cerr << e.what() << "\n";

  const neat::GraphReport& report = e.report();
  std::cerr << "error_code: " << report.error_code << "\n";
  std::cerr << "hint: " << report.repro_note << "\n";
}
```

Useful debug helpers:

```cpp
std::cout << app.describe() << "\n";
std::cout << app.describe_backend() << "\n";
```

- `describe()` prints the public graph summary: endpoints, fragments, and topology.
- `describe_backend()` prints lower-level backend details, useful when debugging generated pipeline
  strings or runtime routing.

For the error-code taxonomy and triage workflow, see [Error codes](/reference/error-codes/).

## Save and load Graph composition

`Graph::save(path)` writes the public graph composition: nodes, endpoint names, explicit endpoint
edges, output options, combine policy, and model-route provenance.

```cpp
app.save("app.graph.json");

neat::Graph loaded = neat::Graph::load("app.graph.json");
neat::Run run = loaded.build();
```

This saves the graph plan, not a running pipeline and not runtime metrics. For runtime metrics, use
Run JSON export.

Model-route provenance matters. A model fragment is more than a list of backend snippets: it carries
input/output names derived from the model archive, route options, and input-route processor metadata
for multi-input models. If a saved graph contains a model fragment, NEAT stores the model archive
path and route options needed to rehydrate it. If the archive is missing on load, NEAT fails with an
actionable error instead of silently building an incomplete route.

## Export and visualize what ran

A `Run` knows both the public graph shape and the lowered runtime shape. It can be exported as a
versioned JSON artifact for CI, debugging, support tickets, or offline visualization.

### Build-time topology snapshot

Use build-time export when you want an artifact as soon as the graph builds:

```cpp
neat::RunOptions opt;
opt.run_export.path = "/tmp/startup.graph_run.json";
opt.run_export.label = "startup";

neat::Run run = app.build(opt);
```

This is an initial topology snapshot. It may contain zero throughput/latency counters because no
samples have run yet.

Python:

```python
opt = pyneat.RunOptions()
opt.run_export.path = "/tmp/startup.graph_run.json"
opt.run_export.label = "startup"

run = app.build(opt)
```

### Post-run snapshot with metrics

Use post-run export after the workload has run or drained:

```cpp
neat::Run run = app.build();
run.push("image", neat::TensorList{image});
auto out = run.pull("classes", 1000);

neat::RunExportOptions export_opt;
export_opt.label = "after_smoke_test";
export_opt.metadata = {{"test_name", "smoke"}};

std::string err;
if (!neat::save_run_json(run, "/tmp/final.graph_run.json", export_opt, &err)) {
  throw std::runtime_error(err);
}
```

Python:

```python
run = app.build()
run.push("image", [image])
out = run.pull("classes", timeout_ms=1000)

export_opt = pyneat.RunExportOptions()
export_opt.label = "after_smoke_test"
export_opt.metadata = [("test_name", "smoke")]

run.save_json("/tmp/final.graph_run.json", export_opt)
```

The exporter snapshots the current run; it does not stop the run. If you need final numbers for a
finite workload, call `run.close_input()` and drain outputs, or call `run.stop()`, before saving.

To include board power telemetry:

```cpp
neat::RunOptions opt;
opt.enable_board_power(/*sample_interval_ms=*/100);

neat::Run run = app.build(opt);
```

The JSON schema is `sima.neat.graph_run` version `1`. The schema lives at
`schemas/graph_run_v1.schema.json` and the CI validator lives at
`tests/perf/tools/graph_run_schema.py`.

Render the artifact without internet access:

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json -o /tmp/final.graph_run.html
```

Choose which view to render:

```bash
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view public
python3 tools/visualize_graph_run.py /tmp/final.graph_run.json --view lowered
```

- `public` shows the graph the user authored: named inputs, outputs, fragments, and `connect(...)`
  edges.
- `lowered` shows what NEAT executed internally: pipeline segments, generated branch/combine stages,
  queues, and runtime edges.

## Common patterns

### Image classification

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(resnet);
app.add(neat::nodes::Output("classes"));
```

### Object detection

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

### RTSP camera to model to app-pulled output

```cpp
neat::nodes::groups::RtspDecodedInputOptions source_opt;
source_opt.url = "rtsp://camera/stream";

neat::Graph app;
app.add(neat::nodes::groups::RtspDecodedInput(source_opt));
app.add(yolo);
app.add(neat::nodes::Output("detections"));
```

### App input to graph-owned UDP output

```cpp
neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::UdpOutput(udp_options));
```

### Branch preview and model path

```cpp
neat::Graph branch = neat::graphs::Branch("image", {"preview", "model_image"});
```

Name `model_image` to match the model route input, or insert an explicit adapter fragment.

### Combine left/right streams

```cpp
neat::Graph pair = neat::graphs::Combine({"left", "right"},
                                         "pair",
                                         neat::CombinePolicy::ByPts);
```

Use `ByPts` when media timestamps are the synchronization key; use `ByFrame` when frame IDs are the
synchronization key.

### GenAI and other stage fragments

GenAI and other non-linear/stage-based capabilities should still enter application code as public
`Graph` fragments and execute through `Graph::build() -> Run`:

```cpp
neat::Graph app;
app.add(genai_fragment);

neat::Run run = app.build();
run.push("prompt", prompt_sample);
auto token = run.pull("tokens", 1000);
```

The exact GenAI fragment factory and sample helper names depend on the installed GenAI package. The
Graph rule is the same: add or connect public fragments, then use named `Run::push(...)` and
`Run::pull(...)`.

## Anti-patterns and gotchas

### Do not use Graph labels as endpoints

Wrong:

```cpp
neat::Graph image("image");
run.push("image", neat::TensorList{tensor}); // Graph label is not an endpoint.
```

Correct:

```cpp
neat::Graph image;
image.add(neat::nodes::Input("image"));
```

### Do not guess model input names

Wrong:

```cpp
left.add(neat::nodes::Input("my_left"));
app.connect(left, model);
```

Correct:

```cpp
for (const auto& name : model.graph().inputs()) {
  std::cout << name << "\n";
}
```

Then name upstream endpoints to match.

### Do not use unnamed push/pull on multi-endpoint graphs

Wrong:

```cpp
run.push(neat::TensorList{left});
run.push(neat::TensorList{right});
```

Correct:

```cpp
run.push("left", neat::TensorList{left});
run.push("right", neat::TensorList{right});
```

### Do not accidentally fan in without a CombinePolicy

Wrong:

```cpp
neat::Graph bundle;
bundle.add(neat::nodes::Output("bundle"));

app.connect(left, bundle);
app.connect(right, bundle); // Ambiguous: how should left/right be synchronized?
```

Correct:

```cpp
neat::Graph bundle = neat::graphs::Combine({"left", "right"},
                                           "bundle",
                                           neat::CombinePolicy::ByFrame);
```

### Do not insert Input/Output in the middle unless you mean a fragment boundary

`Input` and `Output` are public boundary declarations. In reusable fragments that is exactly what
you want. In a purely linear app, adding an extra `Output` in the middle may create a real pullable
sink and backpressure unless that boundary is consumed by another `connect(...)` edge.

### Do not use lower-level runtime graph APIs in application code

Avoid teaching or writing application code with:

```cpp
graph::Graph
graph::GraphRun
graph::build(...)
```

Use:

```cpp
neat::Graph
neat::Run
app.build()
```

## Advanced note: boundary materialization

Named `Input` and `Output` nodes are declarations of a fragment's public contract. They are higher
level than the runtime objects used to move buffers.

Before executable pipeline construction, `Graph::build()` normalizes boundaries:

| Boundary declaration | Materialized when... | Elided when... |
|---|---|---|
| `nodes::Input("name")` | no upstream graph is connected to it, so it must be a public `Run::push("name", ...)` endpoint | an upstream graph feeds it, so it is only an internal fragment parameter |
| `nodes::Output("name")` | no downstream graph consumes it, so it must be a public `Run::pull("name")` endpoint | a downstream graph consumes it, so it is only an internal fragment return value |

Elided does not mean forgotten. The compiler keeps provenance so `describe()`, validation errors,
metrics, and graph-run JSON can still refer back to the user-facing endpoint name.

This prevents reusable fragments from creating hidden appsrc/appsink-style runtime I/O in the middle
of an application. For example:

```cpp
neat::Graph app;
app.connect(camera, route);
app.connect(route, display);
```

The executable data path is `camera -> route body -> display`, not `camera -> route.Input -> route.Output -> display`
with extra physical sinks/sources in the middle.

## API quick reference

### C++

```cpp
// Composition
neat::Graph app("debug_label");
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::Output("classes"));
app.connect(fragment_a, fragment_b);
app.connect("from_endpoint", "to_endpoint");

// Endpoint inspection
auto graph_inputs = app.inputs();
auto graph_outputs = app.outputs();

// Build/run
neat::Run run = app.build();
run.push("image", neat::TensorList{image});
auto out = run.pull("classes", 1000);

// Runtime endpoint inspection
auto run_inputs = run.input_names();
auto run_outputs = run.output_names();

// Validation/debug/export
neat::GraphReport report = app.validate();
std::cout << app.describe() << "\n";
app.save("app.graph.json");
neat::save_run_json(run, "/tmp/app.graph_run.json");
```

### Python

```python
app = pyneat.Graph("debug_label")
app.add(pyneat.nodes.input("image"))
app.add(model)
app.add(pyneat.nodes.output("classes"))

print(app.inputs())
print(app.outputs())

run = app.build()
run.push("image", [image])
out = run.pull("classes", timeout_ms=1000)

print(run.input_names())
print(run.output_names())

app.save("app.graph.json")
run.save_json("/tmp/app.graph_run.json")
```

## Further reading

- [Model programming model](/develop-apps/development-workflow/model)
- [Node programming model: groups and boundaries](/develop-apps/development-workflow/node#boundary-nodes)
- [Tensor and Sample programming model](/develop-apps/development-workflow/core_types)
- [Runtime tuning (Tutorial 015)](/tutorials/015-tune-throughput-and-queues)
- [Diagnostics (Tutorial 011)](/tutorials/011-diagnose-a-pipeline)
- [GStreamer layer](/develop-apps/advanced-concepts/gstreamer_layer)
