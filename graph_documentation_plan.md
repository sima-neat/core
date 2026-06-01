# Graph Documentation Plan

Status: implementation plan only — no user docs changed by this file.
Target reader: application developers using NEAT from C++ or Python.
Target outcome: a clear, progressive user manual that teaches how to build applications with `simaai::neat::Graph` without exposing old/low-level graph internals.

## 0. Goal

Create a user-facing Graph manual that answers:

1. When should I use `Model` directly vs `Graph`?
2. What are `Graph`, `Node`, `Model`, `Input`, `Output`, and `Run`?
3. How do I build the simplest app graph?
4. When do I use `add()` vs `connect()`?
5. How do named inputs/outputs work?
6. How do I inspect and avoid guessing endpoint names?
7. How do I build multi-input, multi-output, branching, and combining apps?
8. How do I validate, debug, save, and visualize a graph run?
9. What mistakes should I avoid?

The manual should be a **teaching doc**, not an internal architecture doc. It should keep the first path simple and progressively introduce power features.

## 1. Where to put it

### Primary file

Rewrite/expand:

```text
docs/concepts/graphs.md
```

Reason:

- It already exists and is already linked from the docs sidebar as the Graph concept page.
- It currently mixes manual content, endpoint materialization details, and export details. We should turn it into the authoritative Graph user manual.

### Supporting links

Update these only if needed after the main doc is written:

```text
docs/doxygen/mainpage.md
docs/getting-started/programming-model/graph.md
README.md
```

Rule: only add small links to the main manual. Do not duplicate the manual.

## 2. Documentation principles

### 2.1 Teach public API only

Use public names:

```cpp
simaai::neat::Graph
simaai::neat::Model
simaai::neat::Run
simaai::neat::nodes::Input
simaai::neat::nodes::Output
simaai::neat::graphs::Branch
simaai::neat::graphs::Combine
simaai::neat::CombinePolicy
```

Do **not** teach users to use:

```cpp
simaai::neat::graph::Graph
simaai::neat::graph::GraphRun
simaai::neat::graph::NodeId
graph::build(...)
```

Those are internal/legacy/lower-level runtime surfaces from the user point of view.

### 2.2 Start from the common path

The first complete example should be:

```cpp
#include <neat.h>

namespace neat = simaai::neat;

int main() {
  neat::Model model("resnet50.tar.gz");

  neat::Graph app;
  app.add(neat::nodes::Input("image"));
  app.add(model);
  app.add(neat::nodes::Output("classes"));

  neat::Run run = app.build();

  neat::Tensor image = /* create or load tensor */;
  run.push("image", neat::TensorList{image});

  auto out = run.pull("classes", 1000);
  if (out) {
    // consume Sample
  }

  run.stop();
}
```

Then explain each line.

### 2.3 Prefer exact endpoint names

The manual must emphasize:

```cpp
Graph g("debug_name");
```

sets a label/debug name only. It does **not** declare a public endpoint.

Public endpoints are declared by:

```cpp
nodes::Input("image")
nodes::Output("classes")
```

or by model route boundary hints from:

```cpp
model.graph().inputs()
model.graph().outputs()
```

### 2.4 Avoid examples that rely on guessing

Bad beginner example:

```cpp
app.connect(left_camera, model);
app.connect(right_camera, model);
```

This can be ambiguous unless the endpoint names match model inputs. Instead, teach:

```cpp
Graph route = model.graph();
for (const auto& name : route.inputs()) {
  std::cout << "model input: " << name << "\n";
}
```

Then name upstream fragments explicitly to match those inputs.

### 2.5 Separate beginner and advanced content

Beginner sections should avoid:

- `Model::RouteOptions`
- `include_input`
- `include_output`
- `expose_all_outputs`
- build-time run export
- endpoint materialization internals
- branch/combine lowering internals

Those go later.

## 3. Proposed final `docs/concepts/graphs.md` structure

## 3.1 Title and short summary

Title:

```md
# Building applications with Graph
```

Opening summary:

```text
Use Model when you want to run one compiled model. Use Graph when you want to build an application around models and nodes: add inputs/outputs, connect reusable fragments, branch streams, combine streams, validate the app, and save/visualize what actually ran.
```

Mental model:

```text
Model = compiled model archive loaded from disk
Node  = one reusable processing step
Graph = application wiring plan
Run   = live execution handle returned by Graph::build()
```

## 3.2 When do I need Graph?

Teach this decision table:

| Goal | Recommended API |
|---|---|
| Run one model on one input | `Model::run()` or `Model::build()` |
| Add input/output nodes around a model | `Graph` |
| Compose model with custom nodes | `Graph::add(...)` |
| Reuse a sub-pipeline | return/pass a `Graph` fragment |
| Route multiple inputs/outputs | named `Input`/`Output` + `connect()` |
| Branch or combine streams | `graphs::Branch`, `graphs::Combine` |
| Save/visualize the executed topology | `save_run_json(run, ...)` |

## 3.3 First Graph: one input, one model, one output

C++ snippet:

```cpp
#include <neat.h>

namespace neat = simaai::neat;

neat::Model model("resnet50.tar.gz");

neat::Graph app;
app.add(neat::nodes::Input("image"));
app.add(model);
app.add(neat::nodes::Output("classes"));

neat::Run run = app.build();
```

Explain:

- `Input("image")` means application code will push data with `run.push("image", ...)`.
- `add(model)` inserts the model route selected from the model archive.
- `Output("classes")` means application code will pull data with `run.pull("classes")`.
- `build()` validates and compiles the whole Graph and returns `Run`.

Python companion:

```python
import pyneat

model = pyneat.Model("resnet50.tar.gz")

app = pyneat.Graph()
app.add(pyneat.nodes.input("image"))
app.add(model)
app.add(pyneat.nodes.output("classes"))

run = app.build()
```

Confirm Python factory names from bindings before finalizing exact spelling. If current docs/tests use `pyneat.nodes.input`, use that. If they expose `pyneat.Input`, use current public spelling.

## 3.4 Running the Graph

C++:

```cpp
run.push("image", neat::TensorList{image});
std::optional<neat::Sample> result = run.pull("classes", 1000);
```

Explain accepted push types:

- `TensorList` for tensors.
- `Sample` when metadata matters: timestamps, stream ID, labels, EOS, text/audio/video semantics.
- `std::vector<cv::Mat>` for image convenience paths.

Mention exact current Run API:

```cpp
run.push(const TensorList&)
run.push("name", const TensorList&)
run.push(const Sample&)
run.push("name", const Sample&)
run.pull(timeout_ms)
run.pull("name", timeout_ms)
run.pull_tensors(...)
run.pull_samples(...)
```

Do not teach single `Tensor` or single Python tensor/sample push if the Python bindings reject single-object API. Use list/batch forms in Python examples.

## 3.5 `build()` vs `build(first_input)`

This is important and should be early.

Teach:

```cpp
Run run = app.build();
```

Use when the graph already declares enough information, or source nodes own input.

Teach:

```cpp
Run run = app.build(neat::TensorList{first_image});
```

Use when the first input should seed shape/caps adaptation.

Explain:

- `build(first_input)` can validate shape/format before streaming starts.
- `RunOptions::startup_preflight` defaults true for seeded build paths and may push/pull the seed once to catch first-sample failures.
- For final runtime metrics, save after actual execution, not immediately after build.

C++ snippet:

```cpp
neat::RunOptions opt;
opt.startup_preflight = true;

neat::Run run = app.build(neat::TensorList{first_image}, neat::RunMode::Async, opt);
```

## 3.6 Graph names are not endpoint names

Make this a bold warning box.

```cpp
Graph g("camera_route");
```

means label/debug name.

It does **not** mean:

```cpp
run.push("camera_route", ...);
```

Correct endpoint declaration:

```cpp
Graph g("camera_route");
g.add(nodes::Input("image"));
g.add(nodes::Output("classes"));
```

## 3.7 Named endpoint inspection

Show both graph-time and run-time inspection.

Before build:

```cpp
auto inputs = app.inputs();
auto outputs = app.outputs();
```

After build:

```cpp
auto inputs = run.input_names();
auto outputs = run.output_names();
```

Explain:

- `Graph::inputs()` / `outputs()` are declared logical endpoints.
- `Run::input_names()` / `output_names()` are what the built runtime accepts.
- Use these to avoid guessing.

## 3.8 Unnamed convenience APIs

Teach after named endpoints.

```cpp
Graph g;
g.add(nodes::Input());
g.add(model);
g.add(nodes::Output());

Run run = g.build();
run.push(TensorList{image});
auto result = run.pull();
```

Rules:

- This is fine for one input and one output.
- For multi-input or multi-output graphs, use names.
- If names are ambiguous, unnamed `push()`/`pull()` fails closed and lists available names.

## 3.9 Models inside Graphs

Teach model-as-fragment:

```cpp
Model yolo("yolov8.tar.gz");

Graph app;
app.add(nodes::Input("image"));
app.add(yolo);
app.add(nodes::Output("boxes"));
```

Explain:

- `Graph::add(model)` inserts the model’s default selected route as a linear fragment.
- It does not force users to call `model.graph()` for simple composition.
- The model route includes preprocess/inference/postprocess stages chosen from the model archive and options.

## 3.10 Inspecting model endpoint names

Teach:

```cpp
Graph route = model.graph();
for (const auto& name : route.inputs()) {
  std::cout << "input: " << name << "\n";
}
for (const auto& name : route.outputs()) {
  std::cout << "output: " << name << "\n";
}
```

Explain:

- Endpoint matching is exact.
- `Input("image_l")` can bind to model input `"image_l"`.
- `Input("my_random_name")` does not implicitly bind to `"image_l"`.

## 3.11 `Model::RouteOptions` for standalone model Graphs

Move this to advanced section.

Show:

```cpp
Model::RouteOptions opt;
opt.include_input = true;
opt.include_output = true;

Graph standalone = model.graph(opt);
Run run = standalone.build();
```

Explain:

- Default `model.graph()` returns a reusable fragment with open endpoints.
- `include_input = true` adds explicit `nodes::Input` boundary nodes.
- `include_output = true` adds explicit `nodes::Output` boundary nodes.
- Use this when you intentionally want a model route to be standalone/runnable by itself.

Explain multi-output behavior:

```cpp
opt.expose_all_outputs = true;
```

- Default is one aggregate logical model output when the route contract says so.
- Set `expose_all_outputs` only for advanced/debug code that needs separate physical outputs.
- If there is only one physical output, this still exposes one output.

## 3.12 `add()` vs `connect()`

Teach with a simple table:

| API | Meaning | Use when |
|---|---|---|
| `add(x)` | append/splice into the current linear chain | `input -> model -> output` |
| `connect(a, b)` | wire endpoint graph fragments | reusable fragments, branching, multiple inputs/outputs |
| `connect("a", "b")` | connect endpoints already declared inside the same graph | building helper fragments |

Examples:

Linear:

```cpp
Graph app;
app.add(nodes::Input("image"));
app.add(model);
app.add(nodes::Output("classes"));
```

Fragment composition:

```cpp
Graph app;
app.connect(camera, route);
app.connect(route, sink);
```

Internal endpoint wiring:

```cpp
Graph route;
route.add(nodes::Input("image"));
route.add(nodes::Output("classes"));
route.connect("image", "classes");
```

Be careful: the last snippet is a conceptual routing helper. Only include if it is genuinely useful and passes build/docs examples. Otherwise keep it for `Branch` / `Combine` helpers.

## 3.13 Reusable Graph fragments

Teach that a function can return a `Graph`:

```cpp
Graph make_classifier(Model& model) {
  Graph g("classifier");
  g.add(nodes::Input("image"));
  g.add(model);
  g.add(nodes::Output("classes"));
  return g;
}
```

Then use it:

```cpp
Graph app;
Graph classifier = make_classifier(model);
app.add(std::move(classifier));
```

For connected fragments, use `connect()` rather than forcing linear splice.

## 3.14 Branching one stream

Use current helper:

```cpp
Graph branch = graphs::Branch("image", {"preview", "model_input"});
```

Explain:

```text
image -> preview
      -> model_input
```

Then show composition with matching names:

```cpp
Graph camera;
camera.add(nodes::Input("image"));

Graph preview;
preview.add(nodes::Output("preview"));

Graph app;
app.connect(camera, branch);
app.connect(branch, preview);
```

If connecting branch to a model, emphasize naming:

```cpp
Graph route = model.graph();
// Choose branch output names to match route.inputs().
```

Do not show a branch-to-model example unless the endpoint names are explicit and correct.

## 3.15 Combining multiple streams

Use current helper:

```cpp
Graph pair = graphs::Combine({"left", "right"}, "stereo", CombinePolicy::ByFrame);
```

Explain policies exactly from code:

- `CombinePolicy::None`: no combining; multiple producers fail closed.
- `CombinePolicy::ByFrame`: match exact `Sample::frame_id`; no PTS fallback.
- `CombinePolicy::ByPts`: match exact `Sample::pts_ns`; no frame-id fallback.

Mention this in plain language:

```text
ByFrame means “give me left and right samples with the same frame number.”
ByPts means “give me samples with the same media timestamp.”
```

Add IDE-friendly note:

```text
If samples do not carry the required key, the combine stage fails with a diagnostic instead of guessing.
```

## 3.16 Sources and sinks

Teach four common endpoint styles.

### App-pushed input

```cpp
g.add(nodes::Input("image"));
run.push("image", TensorList{image});
```

### Graph-owned input source

```cpp
g.add(nodes::RTSPInput(...));
```

or:

```cpp
g.add(nodes::StillImageInput(...));
```

### App-pulled output

```cpp
g.add(nodes::Output("classes"));
auto out = run.pull("classes");
```

### Graph-owned output sink

```cpp
g.add(nodes::UdpOutput(...));
```

or RTSP server mode:

```cpp
RtspServerHandle server = g.run_rtsp(opts);
```

## 3.17 Validation and diagnostics

Teach:

```cpp
GraphReport report = app.validate();
if (!report.error_code.empty()) {
  std::cerr << report.repro_note << "\n";
}
```

Teach build errors:

```cpp
try {
  Run run = app.build();
} catch (const NeatError& e) {
  std::cerr << e.what() << "\n";
  std::cerr << e.report().repro_note << "\n";
}
```

Verify exact `NeatError::report()` signature before final snippet.

Teach describe:

```cpp
std::cout << app.describe() << "\n";
std::cout << app.describe_backend() << "\n";
```

Explain:

- `describe()` = public/human graph summary.
- `describe_backend()` = GStreamer launch string for low-level debugging.

## 3.18 Save/load Graphs

Teach only what the code supports:

```cpp
app.save("app.graph.json");
Graph loaded = Graph::load("app.graph.json");
```

Explain:

- Saves Graph composition/topology/configuration.
- Does not save a running pipeline.
- For runtime metrics, use Run JSON export.

## 3.19 Export and visualize a Run

Use current APIs.

Build-time topology snapshot:

```cpp
RunOptions opt;
opt.run_export.path = "/tmp/startup.graph_run.json";
opt.run_export.label = "startup";

Run run = app.build(opt);
```

Post-run metrics snapshot:

```cpp
RunExportOptions export_opt;
export_opt.label = "after_smoke_test";
export_opt.metadata = {{"test_name", "smoke"}};

std::string err;
if (!save_run_json(run, "/tmp/final.graph_run.json", export_opt, &err)) {
  throw std::runtime_error(err);
}
```

Python:

```python
export_opt = pyneat.RunExportOptions()
export_opt.label = "after_smoke_test"
run.save_json("/tmp/final.graph_run.json", export_opt)
```

Explain:

- `save_run_json` snapshots; it does not stop the run.
- Use `run.close_input()` and drain outputs, or `run.stop()`, before final metrics if you need final numbers.
- Enable board power through `RunOptions::enable_board_power()` when needed.

## 3.20 Common patterns cookbook

Keep examples short and real.

### Pattern A: image classification

```cpp
Graph app;
app.add(nodes::Input("image"));
app.add(resnet);
app.add(nodes::Output("classes"));
```

### Pattern B: object detection

```cpp
Graph app;
app.add(nodes::Input("image"));
app.add(yolo);
app.add(nodes::Output("detections"));
```

### Pattern C: RTSP camera to model to pulled output

```cpp
Graph app;
app.add(nodes::RTSPInput(...));
app.add(yolo);
app.add(nodes::Output("detections"));
```

### Pattern D: app input to UDP/RTSP output

```cpp
Graph app;
app.add(nodes::Input("image"));
app.add(model);
app.add(nodes::UdpOutput(...));
```

### Pattern E: branch preview and model path

Use `graphs::Branch` with explicit endpoint names.

### Pattern F: combine left/right streams

Use `graphs::Combine` with `ByFrame` or `ByPts`.

### Pattern G: GenAI fragment

Only include after verifying exact public API names:

```cpp
auto vlm = std::make_shared<genai::VisionLanguageModel>(...);
Graph vlm_graph = genai::graphs::VisionLanguage(vlm);

Graph app;
app.add(std::move(vlm_graph));
Run run = app.build();
run.push("image", image_sample);
run.push("prompt", prompt_sample);
auto token = run.pull("tokens");
```

Need verify exact model constructor and sample creation before final docs.

## 3.21 Anti-patterns / gotchas

Include a clear final section.

### Do not use Graph labels as endpoints

Wrong:

```cpp
Graph image("image");
run.push("image", ...); // Graph label is not an endpoint.
```

Correct:

```cpp
Graph image;
image.add(nodes::Input("image"));
```

### Do not guess model input names

Wrong:

```cpp
left.add(nodes::Input("my_left"));
app.connect(left, model);
```

Correct:

```cpp
for (auto& name : model.graph().inputs()) {
  std::cout << name << "\n";
}
```

Then name upstream endpoints to match.

### Do not use unnamed push/pull on multi-endpoint graphs

Wrong:

```cpp
run.push(TensorList{left});
run.push(TensorList{right});
```

Correct:

```cpp
run.push("left", TensorList{left});
run.push("right", TensorList{right});
```

### Do not accidentally fan-in without a CombinePolicy

Wrong:

```cpp
Graph out;
out.add(nodes::Output("bundle"));
app.connect(left, out);
app.connect(right, out);
```

Correct:

```cpp
Graph out = graphs::Combine({"left", "right"}, "bundle", CombinePolicy::ByFrame);
```

### Do not teach or use lower-level runtime graph APIs in app docs

Avoid:

```cpp
graph::Graph
graph::GraphRun
graph::build(...)
```

Use:

```cpp
Graph
Run
Graph::build()
```

## 4. Exact API snippets to verify before writing final docs

Before landing the manual, verify each snippet compiles or has a matching test.

### C++ snippets to compile mentally / via small doc test if possible

1. Basic `Graph + Input + Model + Output`.
2. `Run::push("image", TensorList{...})`.
3. `Run::pull("classes", timeout)`.
4. `Graph::inputs()` / `outputs()`.
5. `Run::input_names()` / `output_names()`.
6. `Model::RouteOptions include_input/include_output`.
7. `graphs::Branch`.
8. `graphs::Combine`.
9. `save_run_json`.
10. `RunOptions::run_export`.

### Python snippets to verify from bindings/tests

1. Graph construction spelling.
2. `pyneat.nodes.input("image")` / `pyneat.nodes.output("classes")` exact names.
3. `run.push("image", [tensor_or_sample])`.
4. `run.pull("classes", timeout)`.
5. `run.input_names()` / `run.output_names()`.
6. `pyneat.graphs.branch(...)` / `pyneat.graphs.combine(...)` exact names if exposed.
7. `RunExportOptions` and `run.save_json(...)`.

If a Python helper is not exposed, do not invent it. Use C++ only for that section or add a note.

## 5. Implementation steps

### Step 1 — rewrite `docs/concepts/graphs.md`

Replace the current mixed concept page with the progressive manual described above.

Keep useful existing material, but move it to later sections:

- named boundary materialization rules;
- save/visualization details;
- `Branch` / `Combine` helper details.

### Step 2 — update sidebar/title metadata if needed

Current frontmatter should become:

```md
---
title: Building applications with Graph
description: How to compose Models, Nodes, named inputs/outputs, branches, combines, and Runs with the public Graph API.
sidebar_position: 3
---
```

### Step 3 — add small links from intro docs

Update only if useful:

```text
docs/doxygen/mainpage.md
docs/getting-started/programming-model/graph.md
README.md
```

Do not duplicate examples in all three.

### Step 4 — run docs generation and link check

Run:

```bash
unset SYSROOT PKG_CONFIG_LIBDIR PKG_CONFIG_SYSROOT_DIR PKG_CONFIG_EXECUTABLE \
      PKG_CONFIG LDFLAGS CFLAGS CXXFLAGS CPPFLAGS

cmake -S . -B build-docs-local \
  -DSIMANEAT_BUILD_DOCS=ON \
  -DSIMANEAT_BUILD_SAMPLES=OFF \
  -DSIMANEAT_BUILD_TESTS=OFF \
  -DSIMANEAT_BUILD_TUTORIALS=OFF \
  -DSIMANEAT_BUILD_PYTHON=OFF \
  -DSIMANEAT_REQUIRE_NEAT_RUNTIME_ARTIFACTS=OFF \
  -DSIMANEAT_REQUIRE_LLIMA_ARTIFACTS=OFF

cmake --build build-docs-local --target docs -j$(nproc)
bash tools/generate_api_docs.sh
python3 tools/generate_python_api_docs.py --repo-root .
python3 tools/generate_tutorial_docs.py --repo-root .
python3 tools/expand_code_tabs.py --src docs --dst build-docs-local/docs-expanded
rm -rf website/.docusaurus website/build
DOCS_PATH="$PWD/build-docs-local/docs-expanded" npm --prefix website run build
DOCS_LINK_SITE_DIR="$PWD/website/build" DOCS_LINK_START_PATHS=all bash scripts/ci/check_docs_links.sh
```

### Step 5 — avoid committing generated category churn

If docs generation modifies only category ordering files unexpectedly, inspect before committing.

Expected final source changes should be mostly:

```text
docs/concepts/graphs.md
```

plus any small intentional link changes.

## 6. Acceptance criteria

The Graph manual is ready when:

- [ ] A new user can understand when to use `Model` vs `Graph`.
- [ ] The first example uses only stable public API.
- [ ] `add()` and `connect()` are explained without teaching low-level graph internals.
- [ ] `Graph("name")` vs `Input("name")` / `Output("name")` is unmistakable.
- [ ] Named inputs/outputs and exact endpoint matching are explained.
- [ ] `build()` vs `build(first_input)` is explained.
- [ ] Multi-input/multi-output models are covered through endpoint inspection, not guessing.
- [ ] `Branch` and `Combine` are covered with policy semantics.
- [ ] Source-owned and app-pushed inputs are both covered.
- [ ] App-pulled and graph-owned outputs are both covered.
- [ ] Validation, `describe()`, save/load, and run JSON export are covered.
- [ ] Anti-patterns warn against the failure modes we have seen in tests.
- [ ] Docs site builds.
- [ ] Linkinator passes.

## 7. Risks / things not to invent

Do not invent APIs for the manual.

Before finalizing snippets, verify exact spelling for:

- Python node factories.
- Python `graphs::Branch` / `graphs::Combine` exposure.
- GenAI public constructors/sample helpers.
- `NeatError::report()` accessor syntax.
- Whether current Python docs prefer `pyneat.TensorList` or plain Python lists.

If an API is not currently exposed, write C++ only or say “C++ only for now.”

## 8. Suggested final manual tone

Use plain language:

- “Graph is your application wiring diagram.”
- “Input and Output are the public doors of the graph.”
- “Build turns the diagram into a live Run.”
- “Named endpoints prevent guessing.”
- “NEAT fails closed when routing is ambiguous.”

Avoid internal language in beginner sections:

- “ExecutionGraphPlan”
- “RunCore”
- “appsink/appsrc materialization”
- “runtime graph substrate”
- “port handles / NodeId”

Those may appear only in one short advanced note if needed.
