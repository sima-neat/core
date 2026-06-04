# Model Fix Implementation Plan

Status: planning document only.  
Scope: `Model`, `Graph::add(Model)`, `nodes::groups::Infer`, Python bindings, docs/tutorials, and model-archive/runtime correctness regressions.  
Source baseline inspected: `core_graph_changes` branch `onat_handover_tmp` around 2026-06-03.  
Primary source investigation file: `Model_fix.md`.

This document is intentionally implementation-heavy. It is meant to be the checklist a developer can follow to turn the Model review into small, reviewable PRs without introducing route-planner magic or broad rewrites.

---

## Review revisions — 2026-06-04 (adversarial code-verified pass)

An adversarial review verified every "current behavior / current bug" premise in this plan against the actual source. Net result: the plan is **execute-with-cuts** — roughly a quarter of it fixes real, code-confirmed bugs; the rest is scaffolding that was trimmed. Changes applied below:

- **CUT Ch10 + Ch14 (BoxDecode adapter-only metadata test + fix).** Ch14 fixes a *provably-shadowed* branch (`Model.cpp:4229-4238` early-returns `nullopt` for `boxdecode_selected && graph_family != Preproc`, shadowing the `!rp.enabled` adapter branch), but reaching that branch requires `family == Disabled` (bare-tensor passthrough into BoxDecode) — and no real compiled model demonstrating that topology has been exhibited. Ch10's test is doubly dead: it asserts `preprocess_meta.has_value()` via `input_appsrc_options_list()` (`Model.cpp:6518-6541`), but `preprocess_meta` is only ever set in `build_pipeline_nodes` (`Model.cpp:5753`), so the assertion is false **before and after** the fix; and its `quanttess` fixture forces `family == QuantTess` (`InputPlanner.cpp:763-765`), hitting the `rp.enabled`-true branch, never the `!rp.enabled` branch Ch14 unshadows. **Both are deferred until a real failing model is exhibited.** See the marked sections.
- **Scope honesty (Goal 3 / Risk 5).** This plan does NOT address the live BoxDecode **pose/seg decode** failures and never claimed to (pose unification is explicitly deferred, and "no magic inference in runtime plugins" is a non-goal). The Ch14 item is a metadata-template **reachability nit**, orthogonal to the decode garbage. The real live failures are a **separate, higher-priority workstream** (see the new "Out of scope" note in Chapter 1) and must NOT be considered closed by this plan.
- **Ch13 (preproc materialization):** keep Implementation **A** only; Implementation **B** is cut as dead weight. The real defect is broader than a self-link — **every pre-region currently receives the same `upstream_name` (no region-to-region chaining)** (`build_preprocess_nodes_impl`, `Model.cpp:5491-5501`). Ch13 must add the missing `infer_upstream` recompute AND a **multi-region preproc regression test** before landing; it is the riskiest core change.
- **PR4 (`NeatError`):** trim the maximalist "wrap every public boundary" framing to the **Model-only** scope actually implemented, and add a **prerequisite `std::invalid_argument` caller audit** (5+ throw sites incl. `Model.cpp:1751,1754,2045,2121`) before flipping public-boundary exception types.
- **Ch15 (archive collision):** gate behind an archive-in-the-wild audit; land as a **loud warning first**, hard-reject only after the audit (Risk 4).
- **CUT Ch27** (speculative `ModelOutputSemanticSpec` / `semantic_output_specs()` — self-deferred API surface, not a fix).
- **Highest-value, lowest-risk, fully-verified deliverable = PR1 (docs/tutorials truth pass).** Land it first and standalone.

Revised landing order is in Chapter 3.

---

## Chapter 0 — User-facing style rule

### Decision

Use a short namespace alias in user-facing C++ examples instead of repeating `simaai::neat::...` everywhere.

Preferred style:

```cpp
namespace neat = simaai::neat;

neat::Model model("/models/yolov8.tar.gz");
neat::TensorList result =
    model.run(neat::TensorList{input_tensor}, /*timeout_ms=*/2000);
```

Reason:

- Short enough for docs and tutorials.
- Avoids global `using namespace simaai::neat;` in snippets.
- Keeps the real namespace visible to users.

For tiny tutorial `main()` examples, this is also acceptable inside function scope:

```cpp
namespace neat = simaai::neat;
```

Do **not** introduce new public aliases in source just for docs.

---

## Chapter 1 — Goals and non-goals

### Goals

1. Make the public `Model` story coherent:
   - `Model` is the high-level full model route abstraction.
   - `Model::run(...)` is the simplest synchronous API.
   - `Model::build(...)` returns a reusable `Model::Runner`.
   - `Graph::add(model)` is the preferred way to put a full model route inside a user graph.
   - `model.graph(...)` is the advanced route-fragment escape hatch.
   - `nodes::groups::Infer(...)` means the inference-stage fragment, not full pre+infer+post.

2. Make public setup/build/run errors debuggable:
   - Model archive failures should become structured `NeatError`s.
   - Metadata parse failures should become structured `NeatError`s.
   - Public Model input misuse should become structured `NeatError`s.
   - Python Model input misuse should become `pyneat.NeatError`.

3. Fix verified correctness bugs:
   - Preprocess route materialization self-links and fails to chain pre-regions (every region gets the same upstream). **(verified — Ch13)**
   - Model archive extraction can flatten different archive paths onto the same output path. **(verified — Ch15)**
   - ~~Adapter-only routes feeding BoxDecode can skip needed preprocess metadata.~~ **(DEFERRED — Ch14: provably-shadowed but unproven-reachable; orthogonal to the live decode failures. Do not land until a real failing model is exhibited.)**

4. Improve Python parity for stable `Model` APIs.

### Non-goals

- Do not collapse the route planner in the first pass.
- Do not redefine `Infer()` as full route.
- Do not add a new `Full()` API; existing `Model::run`, `Model::build`, `Graph::add(model)`, and `model.graph(...)` are enough.
- Do not add magic inference in runtime plugins where MPK contracts already contain the truth.
- Do not change direct extracted-directory behavior beyond documentation/optional diagnostics unless explicitly approved.

### Out of scope — the live BoxDecode pose/seg decode failures (separate workstream)

This plan operates only in `core_graph_changes` and does **not** fix the live, validated decode failures (`yolov8n/yolo11n/yolo26n-pose` crash; bf16 under-detection). Those root causes live mostly in the gst plugins/kernel (`clean_internals`) and the BoxDecode static-contract extractor, and are tracked separately:

- Missing `YoloV26Pose` enum + `yolo26-pose` kernel token (verified absent: `include/pipeline/BoxDecodeType.h` jumps `YoloV8Pose=8 → YoloV26=17`) → yolo26-pose is structurally unreachable.
- `compute_pose()` hardcodes `int8` reads + computes `num_pose_points = channels/3` over **physical** (C16-padded) depth → BF16 broken and `21 > MAX_NUM_POSE_POINTS(17)` overflow (`clean_internals/.../genericboxdecode/src/boxdecode.cpp`). The surgical fix reads logical `slice_ch/3` and clamps.
- `num_classes` is a **config-time gate** (kernel `score_tensor->channels % num_classes`); core never emits it and the plugin never returns 1 for pose.
- Physical-vs-logical channel depth in `src/pipeline/internal/sima/BoxDecodeStaticContractExtractor.cpp`.

**Do not consider that ticket closed by anything in this document.**

---

## Chapter 2 — Current source findings to preserve in review notes

### Files inspected

Core Model/API:

- `include/model/Model.h`
- `src/model/Model.cpp`
- `src/model/ModelPack.cpp`
- `src/model/ModelArchiveLoader.cpp`
- `src/model/InputPlanner.cpp`
- `src/model/RoutePlanner.cpp`
- `src/pipeline/graph/GraphModel.cpp`
- `src/pipeline/graph/Graph.cpp`
- `include/nodes/groups/ModelGroups.h`
- `src/nodes/groups/ModelGroups.cpp`
- `python/src/module.cpp`
- `python/pyneat/_wrappers.py`
- `python/tests/test_api_surface.py`

### Confirmed current behavior

1. `Graph::add(const Model&)` currently does:

```cpp
Graph& Graph::add(const Model& model) {
  Model::RouteOptions opt;
  opt.include_input = false;
  opt.include_output = false;
  return add(model.graph(opt));
}
```

So `Graph::add(model)` is already the right high-level composition path.

2. `nodes::groups::Infer(const Model&)` currently returns only `model.inference()`:

```cpp
simaai::neat::Graph Infer(const simaai::neat::Model& model) {
  return model.inference();
}
```

The docs saying it chains preprocess + MLA + postprocess are stale.

3. `Model::run(...)` now caches a runner after first use:

```cpp
std::lock_guard<std::mutex> lock(impl_->sync_mu);
if (!impl_->sync_ready) {
  impl_->sync_runner = build(inputs);
  impl_->sync_ready = true;
}
return impl_->sync_runner.run(inputs, timeout_ms);
```

So old wording that `Model::run` rebuilds every call is no longer accurate.

4. C++ Model introspection exists:

- `input_specs()`
- `output_specs()`
- `compiled_batch_size()`
- `resolved_preprocess_plan()`
- `preprocess_requirements()`
- `info()`
- `input_appsrc_options_list()`

Python only exposes a subset.

5. Public docs/header promise `NeatError` for many Model paths, but implementation still leaks raw exceptions at important boundaries.

---

## Chapter 3 — Patch/PR sequencing

Recommended split (revised after the 2026-06-04 review — leaner order, only code-verified items):

1. **PR 1 — Documentation truth and tutorial corrections** *(land first, standalone)*
   - Low risk, no runtime behavior change, fully verified, highest user value. Ch4–8.

2. **PR 2 (slim) — Archive basename-collision fix + test**
   - Real data-loss bug, contained blast radius. Ch11 test + Ch15 fix, **landed as a warning first** (hard-reject only after an archive-in-the-wild audit). Add Ch12 Model error-contract tests here as the regression net for PR4.
   - (Original PR2's Ch9/Ch10 split: keep Ch9 preproc test with Ch13; **Ch10 is cut** — see Chapter 10.)

3. **PR 3 — Model `NeatError` normalization** *(was PR4)*
   - Model-only scope (not "every boundary"). **Prerequisite:** `std::invalid_argument` caller audit (`Model.cpp:1751,1754,2045,2121`, …). Document the `pyneat.Model`-vs-`Graph` `NeatError` split. Ch16–22.

4. **PR 4 — Preproc upstream/chaining fix** *(was Ch13, the riskiest core change — land only after its test exists)*
   - Implementation A only. Must include the `infer_upstream` recompute **and** a multi-region preproc regression test (Ch9). Ch13.

5. **PR 5 — Python parity bindings**
   - Stable Model introspection APIs + runner diagnostics. Ch23–25. (Audit current bindings first — some may already exist.)

**Cut entirely:** Ch10 (dead test), Ch14 (unproven-reachable dead code — defer until a real failing model is exhibited), Ch27 (speculative API), Chapter 0 namespace-style rule (bikeshed; fold into PR1 if anything).

**Not in this plan (separate, higher-priority ticket):** the live BoxDecode pose/seg decode fix — see Chapter 1 "Out of scope".

This order keeps review surface small and makes regressions easier to isolate.

---

# PR 1 — Documentation truth and tutorial corrections

---

## Chapter 4 — Update `include/model/Model.h`

### Current problem

The header example shows a single tensor being passed directly:

```cpp
sima::Model model("/models/yolov8.tar.gz");
auto result = model.run(input_tensor);   // shortest path
```

But current C++ API takes `TensorList`, `Sample`, or `std::vector<cv::Mat>`.

### Implementation detail

Edit the top-level class documentation in `include/model/Model.h`.

Recommended replacement snippet:

```cpp
 * @code
 *   namespace neat = simaai::neat;
 *
 *   neat::Model model("/models/yolov8.tar.gz");
 *   neat::TensorList outputs =
 *       model.run(neat::TensorList{input_tensor}, /*timeout_ms=*/2000);
 * @endcode
```

Also adjust the architecture wording.

Suggested text:

```cpp
 * `Model` is the user-facing wrapper around a compiled model archive (`.tar.gz`). It loads the
 * file, extracts and validates the manifest, runs the route planner, and exposes ready-to-use
 * `Graph` fragments plus convenience `run()` and `build()` methods.
 *
 * `Model` lowers to the same route fragments used by the public `Graph` API. New users start
 * with `Model::run(...)`. Applications that need graph-level orchestration should normally use
 * `Graph::add(model)`. Use `model.graph(...)` when advanced code intentionally needs the reusable
 * route fragment itself.
```

### Acceptance criteria

- No C++ example in `Model.h` implies single `Tensor` is accepted directly.
- Docs identify `Graph::add(model)` as the normal graph-composition path.
- `model.graph(...)` is described as advanced/fragment-level.

---

## Chapter 5 — Update `include/nodes/groups/ModelGroups.h`

### Current problem

Header says `Infer` is a combined fragment:

```cpp
 * Graph — preprocess, MLA, postprocess, and the combined `Infer` fragment that chains
 * them together.
```

But implementation returns `model.inference()`.

### Implementation detail

Replace the file header with:

```cpp
/**
 * @file
 * @ingroup nodes_groups
 * @brief Model-stage Graph fragments: preprocess, MLA inference, and postprocess helpers.
 *
 * These helpers expose individual stage fragments from a compiled `Model`. `Infer(...)` means
 * the inference-stage fragment today: the MLA route portion. It is intentionally not the full
 * pre+infer+post model route.
 *
 * For a full model route use:
 *   - `Model::run(...)`
 *   - `Model::build(...)`
 *   - `Graph::add(model)`
 *   - `model.graph(...)` for advanced fragment composition
 *
 * @see Model
 */
```

For `Infer` declarations, use:

```cpp
/**
 * @brief Build the inference-stage Graph fragment from an already-parsed `Model`.
 *
 * This is not the full model route. It currently maps to `model.inference()`; future
 * multi-artifact inference routes may expand this stage beyond one MLA node.
 *
 * @param model The parsed model.
 * @ingroup nodes_groups
 */
simaai::neat::Graph Infer(const simaai::neat::Model& model);
```

And for path overloads:

```cpp
/**
 * @brief Build the inference-stage Graph fragment from a model archive.
 *
 * This is not the full model route. Use `Model::run`, `Model::build`, `Graph::add(model)`,
 * or `model.graph(...)` for full model execution.
 */
simaai::neat::Graph infer(const std::string& tar_gz, const InferOptions& opt);
```

### Acceptance criteria

- No docs imply `Infer` runs preprocess/postprocess.
- Full route APIs are listed explicitly.

---

## Chapter 6 — Update `docs/doxygen/mainpage.md`

### Current problem

The main page encourages:

```cpp
graph.add(model.graph());
```

This works, but it is not the cleanest public story. The preferred pattern is `graph.add(model)`.

### Implementation detail

Replace snippets like this:

```cpp
sima::Model model("/models/yolov8.tar.gz");
...
graph.add(model.graph());
```

with:

```cpp
namespace neat = simaai::neat;

neat::Model model("/models/yolov8.tar.gz");

neat::Graph graph;
graph.add(neat::nodes::Input("image"));
graph.add(model);  // full model route: preprocess + MLA + postprocess as needed
graph.add(neat::nodes::Output("result"));
```

Add a short note:

```md
Prefer `graph.add(model)` for normal app composition. Use `model.graph(...)` when you need the
route fragment object itself, for example to inspect or reuse a stage range deliberately.
```

### Acceptance criteria

- Main page uses namespace alias.
- Main page leads users to `Graph::add(model)`.
- `model.graph(...)` remains documented as advanced.

---

## Chapter 7 — Update tutorial 001: `run_your_first_model`

Files:

- `tutorials/001_run_your_first_model/README.md`
- `tutorials/001_run_your_first_model/run_your_first_model.py`
- `tutorials/001_run_your_first_model/run_your_first_model.cpp`

### Current problem

README says Python `model.run(...)` returns a `Sample`. For tensor/numpy input, Python binding returns a tensor list.

Current Python:

```python
sample = model.run([image], timeout_ms=2000)
top1 = int(np.argmax(sample.tensor.to_numpy().reshape(-1)))
```

### Implementation detail — Python

Change to:

```python
outputs = model.run([image], timeout_ms=2000)
top1 = int(np.argmax(outputs[0].to_numpy().reshape(-1)))
```

If `outputs[0]` is not supported because the binding returns `TensorList` with a different list API, use the existing successful pattern in Python tests/tutorials. Do not reintroduce `sample.tensor` for tensor inputs.

### Implementation detail — C++

Add alias:

```cpp
namespace neat = simaai::neat;
```

Replace long types in core logic:

```cpp
neat::Model model(model_path, build_options(size));
cv::Mat input = image.empty() ? cv::Mat(size, size, CV_8UC3, cv::Scalar(99, 99, 99))
                              : load_rgb(image, size);
neat::TensorList outputs = model.run(std::vector<cv::Mat>{input}, /*timeout_ms=*/2000);
```

Rename helper argument from `out` if desired:

```cpp
int top1_from_output(const neat::TensorList& outputs) {
  if (outputs.empty())
    throw std::runtime_error("no tensor output");
  const neat::Mapping m = outputs.front().map_read();
  ...
}
```

### Implementation detail — README

Replace:

```md
- `model.run(input, timeout_ms)` — synchronous inference; returns a `Sample` with the model's output tensor.
```

with:

```md
- `model.run([input], timeout_ms)` — synchronous inference.
  - Tensor/image inputs return a list of output tensors.
  - Sample inputs return output Samples.
  - Python intentionally takes a list/tuple even for one input: `model.run([image])`.
```

### Acceptance criteria

- Python tutorial matches actual return contract.
- C++ tutorial uses namespace alias.
- README explicitly shows list/tuple input for Python.

---

## Chapter 8 — Update tutorial 013: `embed_model_inside_graph`

Files:

- `tutorials/013_embed_model_inside_graph/README.md`
- `tutorials/013_embed_model_inside_graph/embed_model_inside_graph.cpp`
- `tutorials/013_embed_model_inside_graph/embed_model_inside_graph.py`

### Current status

This tutorial already mostly uses the right pattern:

```python
graph.add(model)
```

C++ also uses `graph.add(model)`.

### Implementation detail

Only improve C++ readability with namespace alias:

```cpp
namespace neat = simaai::neat;

neat::Model model(model_path);

neat::Graph graph;
graph.add(neat::nodes::Input("image"));
graph.add(model);
graph.add(neat::nodes::Output("result"));
```

Update README concept text to make the distinction explicit:

```md
`graph.add(model)` appends the model's full route fragment: preprocessor, MLA inference, and
postprocessor stages as required by the MPK contract and `Model::Options`.

Use `model.graph(...)` only when you intentionally need the fragment object for advanced
composition or inspection.
```

### Acceptance criteria

- Tutorial remains graph-composition-focused.
- It explicitly says `graph.add(model)` is full route.
- C++ snippet uses alias.

---

# PR 2 — Add regression tests before logic fixes

---

## Chapter 9 — Add preproc materialization regression test

### File

Create:

- `tests/unit_testing/unit_model_route_materialization_test.cpp`

Or extend:

- `tests/unit_testing/unit_model_stage_fragments_test.cpp`

A new file is cleaner because it separates materialization bugs from stage-fragment API behavior.

### Test fixture

Use `sima_test::make_strict_model_archive_fixture(...)`, following existing style in `unit_model_stage_fragments_test.cpp`.

Fixture shape:

```json
{
  "pipelines": [{
    "sequence": [
      {
        "sequence_id": 1,
        "name": "preproc_0",
        "pluginId": "processcvu",
        "configPath": "0_preproc.json",
        "processor": "CVU",
        "kernel": "preproc",
        "input": "decoder"
      },
      {
        "sequence_id": 2,
        "name": "mla_0",
        "pluginId": "processmla",
        "configPath": "0_process_mla.json",
        "processor": "MLA",
        "kernel": "infer",
        "input": "preproc_0"
      }
    ]
  }]
}
```

### Test 1 — preproc must not self-link

```cpp
RUN_TEST("unit_model_route_materialization_preproc_upstream_test", ([] {
  namespace neat = simaai::neat;

  const auto fixture = make_preproc_mla_fixture("model_preproc_upstream");
  neat::Model model(fixture.tar_path);

  neat::Model::RouteOptions opt;
  opt.include_input = false;
  opt.include_output = false;

  const std::string backend = model.graph(opt).describe_backend(false);

  require_contains(backend, "neatprocesscvu", "route should include preproc");
  require_contains(backend, "neatprocessmla", "route should include MLA");

  require(backend.find("preproc_0") != std::string::npos,
          "backend should mention preproc stage");

  require(backend.find("input=(string)preproc_0") == std::string::npos &&
              backend.find("upstream-name=(string)preproc_0") == std::string::npos &&
              backend.find("input-buffers=(string)preproc_0") == std::string::npos,
          "preproc must not point to itself as upstream");
}));
```

Adjust string checks to actual rendered backend syntax. The purpose is to lock the invariant, not exact property names.

### Test 2 — multi pre-region chain

If fixtures can express `cast -> tess -> mla` or `quant -> tess -> mla`, add a second test:

```cpp
RUN_TEST("unit_model_route_materialization_pre_regions_chain_test", ([] {
  namespace neat = simaai::neat;

  const auto fixture = make_multi_prestage_fixture("model_pre_regions_chain");
  neat::Model model(fixture.tar_path);

  const std::string backend = model.graph().describe_backend(false);

  require_contains(backend, "quant", "route should include first pre region");
  require_contains(backend, "tess", "route should include second pre region");

  // Invariant: second pre-stage consumes first pre-stage, MLA consumes final pre-stage.
  require_contains(backend, "quant", "first pre-stage present");
  require_contains(backend, "tess", "second pre-stage present");
}));
```

If exact stage syntax is hard to assert, inspect `Graph` linear nodes if test helpers expose them; otherwise start with a self-link test.

### CMake

Add target to `tests/CMakeLists.txt` near other model unit tests:

```cmake
unit_model_route_materialization_test
```

And add to strict/unit labels where `unit_model_stage_fragments_test` appears.

### Expected initial result

This may fail before PR 3 if self-link is present. That is desired.

---

## Chapter 10 — Add BoxDecode adapter-only metadata regression test

> **CUT (2026-06-04 review).** This test cannot exercise what it claims, in two independent ways:
> 1. It asserts `inputs.front().preprocess_meta.has_value()` reached via `input_appsrc_options_list()` (`Model.cpp:6518-6541`), but `preprocess_meta` is **only** populated inside `build_pipeline_nodes` (`Model.cpp:5753`). The list API never sets it → the assertion is false **before and after** the Ch14 fix.
> 2. The proposed `quanttess` fixture forces `enabled == true` / `family == QuantTess` (`InputPlanner.cpp:763-765`), so it hits the `rp.enabled`-true branch (`Model.cpp:4268`), never the `!rp.enabled` (`family == Disabled`) branch that Ch14 unshadows.
>
> A green version of this test would falsely certify the Ch14 path as guarded. **Do not implement.** If Ch14 is ever revived, write a test that drives `build_pipeline_nodes` with a real `family == Disabled + boxdecode_selected` model. The original draft below is retained for reference only.

### File

Preferred:

- `tests/unit_testing/unit_model_boxdecode_metadata_test.cpp`

Alternative:

- extend `unit_boxdecode_render_manifest_from_model_test.cpp`

### Current bug hypothesis

`make_preprocess_meta_template(...)` has:

```cpp
if (plan.session_route_plan.boxdecode_selected &&
    rp.graph_family != PreprocessGraphFamily::Preproc) {
  return std::nullopt;
}
```

But below it has intended adapter-only logic:

```cpp
if (!rp.enabled) {
  // Adapter-only ingress may still feed boxdecode...
  return meta;
}
```

The lower branch is unreachable for non-Preproc BoxDecode.

### Test fixture

Construct a route like:

```text
quanttess_0 -> mla_0 -> boxdecode_0
```

No actual `preproc` stage.

Include enough MPK/static contract JSON for planner to select BoxDecode. Reuse fixture patterns from existing BoxDecode tests.

### Test body

```cpp
RUN_TEST("unit_model_boxdecode_adapter_only_preprocess_meta_test", ([] {
  namespace neat = simaai::neat;

  const auto fixture = make_quanttess_mla_boxdecode_fixture("boxdecode_adapter_only_meta");

  neat::Model::Options opt;
  opt.decode_type_option = neat::BoxDecodeTypeOption::YoloV8Logit;
  opt.score_threshold = 0.25;
  opt.nms_iou_threshold = 0.45;
  opt.top_k = 100;

  neat::Model model(fixture.tar_path, opt);

  const auto inputs = model.input_appsrc_options_list(/*tensor_mode=*/true);
  require(!inputs.empty(), "model should expose at least one input appsrc option");

  require(inputs.front().preprocess_meta.has_value(),
          "adapter-only BoxDecode route must carry preprocess runtime metadata");

  const auto& meta = *inputs.front().preprocess_meta;
  require(meta.enabled, "preprocess meta should be enabled for BoxDecode");
  require(meta.target_width > 0 && meta.target_height > 0,
          "BoxDecode preprocess meta should carry model input geometry");
}));
```

### Acceptance criteria

- Test fails before metadata fix if branch is truly unreachable.
- Test passes after the surgical branch fix.

---

## Chapter 11 — Add archive extraction collision regression test

### File

Extend:

- `tests/unit_testing/unit_model_archive_loader_test.cpp`

### Current issue

`validate_archive(...)` rejects duplicate archive paths, but extraction flattens paths by basename:

```cpp
if (entry.entry_class == EntryClass::Json) {
  return package_root / "etc" / name;
}
```

So these are unique archive entries:

```text
a/config.json
b/config.json
```

but both extract to:

```text
etc/config.json
```

### Test helper

If existing archive fixture tools can generate arbitrary tar entries, use them. Otherwise add a minimal helper in the test file using existing test utilities.

### Test body

```cpp
require_model_archive_error(
    [&]() {
      const auto fixture = sima_test::make_model_archive_fixture(
          "archive_extract_collision",
          {
              {"etc/pipeline_sequence.json", valid_minimal_pipeline_sequence_json()},
              {"a/config.json", "{}"},
              {"b/config.json", "{}"},
              {"share/model.elf", "dummy"},
          });
      (void)ModelArchiveLoader::inspect(fixture.tar_path);
    },
    ModelArchiveErrorClass::InvalidArchive,
    "archive entries that flatten to the same extraction path should be rejected");
```

If `inspect(...)` should remain purely archive-structure validation and not extraction-layout validation, put the check in `extract(...)` instead and test `ModelArchiveLoader::extract(...)`. My recommendation: validate during `inspect(...)` because the archive is not safe to extract without this property.

### Acceptance criteria

- Collision archive is rejected before any file overwrite.
- Error message names both source paths and the flattened destination.

---

## Chapter 12 — Add Model error contract tests

### File

Create:

- `tests/unit_testing/unit_model_error_contract_test.cpp`

### Helper

```cpp
namespace {

template <typename Fn>
void expect_neat_error(Fn&& fn, const std::string& expected_code,
                       const std::string& context) {
  try {
    fn();
  } catch (const simaai::neat::NeatError& e) {
    const auto& report = e.report();
    require(report.error_code == expected_code,
            context + ": unexpected error_code='" + report.error_code + "'");
    require(!report.repro_note.empty(), context + ": repro_note should not be empty");
    return;
  }
  throw std::runtime_error(context + ": expected NeatError");
}

} // namespace
```

### Test cases

1. Bad archive path or unsupported extension:

```cpp
expect_neat_error(
    [&] { neat::Model model("/tmp/does_not_exist.tar.gz"); },
    neat::error_codes::kModelArchive,
    "missing model archive should map to io.model_archive");
```

2. Empty tensor list:

```cpp
const auto fixture = make_minimal_valid_fixture("model_error_empty_build");
neat::Model model(fixture.tar_path);

expect_neat_error(
    [&] { (void)model.build(neat::TensorList{}); },
    neat::error_codes::kModelInput,
    "empty tensor build should map to misconfig.model_input");
```

3. Malformed metadata:

Use direct extracted directory fixture if easier to mutate `metadata.json` after extraction.

```cpp
expect_neat_error(
    [&] { (void)model.metadata(); },
    neat::error_codes::kModelMetadata,
    "malformed metadata should map to io.model_metadata");
```

### Acceptance criteria

- Tests are written before PR 4.
- Initially may fail with raw exceptions.

---

# PR 3 — Surgical correctness fixes

---

## Chapter 13 — Fix preproc upstream/chaining

> **Review revisions (2026-06-04).** This is the **only core data-flow correctness bug** in the plan and the riskiest change.
> - Use **Implementation A** only. **Implementation B is cut** (more complex, not selected — dead weight).
> - The defect is **broader than a self-link**: `build_preprocess_nodes_impl` (`Model.cpp:5491-5501`) passes the **same** `upstream_name` to every region, so multi-region preproc has **no region-to-region chaining at all**. Implementation A as drafted only fixes the self-link — it must **also** add the per-region `infer_upstream` recompute so region N+1 chains off region N.
> - **Do not land without a multi-region preproc regression test** (Ch9, Test 2 "multi pre-region chain") proving the chained upstreams. Until that test exists and passes, hold this chapter.

### File

- `src/model/Model.cpp`

### Current problematic logic

In `build_pipeline_nodes(...)`:

```cpp
std::string upstream =
    popt.upstream_name.empty() ? (pre_name.empty() ? "decoder" : pre_name) : popt.upstream_name;
```

If preproc exists and no explicit upstream, this sets `upstream = pre_name`. Then preproc can be built with its own name as upstream.

In `build_preprocess_nodes_impl(...)`, each pre-region gets the same `upstream_name`:

```cpp
nodes.push_back(build_preprocess_node_from_region(model, pack, plan, input, sync, region,
                                                  stage_name, upstream_name));
```

This does not chain multiple pre-regions.

### Implementation A — chain pre-regions

Replace the loop in `build_preprocess_nodes_impl(...)` with:

```cpp
std::string current_upstream = upstream_name.empty() ? std::string("decoder") : upstream_name;

for (std::size_t i = 0; i < pre_regions.size(); ++i) {
  const auto& region = pre_regions[i];
  const std::string stage_name = region_element_name(i, region.op_kind);
  if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
    std::fprintf(stderr,
                 "[typed-adapter] pre_region[%zu] kind=%d op=%d members=%zu "
                 "stage_name=%s upstream=%s\n",
                 i, static_cast<int>(region.kind), static_cast<int>(region.op_kind),
                 region.member_plugin_indices.size(), stage_name.c_str(),
                 current_upstream.c_str());
  }
  nodes.push_back(build_preprocess_node_from_region(model, pack, plan, input, sync, region,
                                                    stage_name, current_upstream));
  current_upstream = stage_name;
}
```

### Implementation B — separate route ingress from infer upstream

In `build_pipeline_nodes(...)`, replace generic `upstream` computation with separate names:

```cpp
std::string route_ingress_upstream =
    popt.upstream_name.empty() ? std::string("decoder") : popt.upstream_name;

if (!include_preprocess_stage && popt.include_input && popt.upstream_name.empty() &&
    !src_opt.buffer_name.empty()) {
  route_ingress_upstream = src_opt.buffer_name;
}

std::string infer_upstream = route_ingress_upstream;

if (include_preprocess_stage) {
  const std::string pre_upstream =
      (popt.include_input && !src_opt.buffer_name.empty()) ? src_opt.buffer_name
                                                           : route_ingress_upstream;
  auto pre_nodes =
      build_preprocess_nodes_impl(model, pack, plan, input, pre_name, pre_upstream, sync);
  nodes.insert(nodes.end(), pre_nodes.begin(), pre_nodes.end());

  infer_upstream = resolved_pre_stage_name(pack, plan);
  if (infer_upstream.empty() && !pre_nodes.empty()) {
    infer_upstream = pre_name.empty() ? route_ingress_upstream : pre_name;
  }
}

auto infer_nodes = pack.infer_block(
    infer_upstream, make_stage_lineage_binding(model, internal::ModelLineageStageRole::Infer));
```

### Important review note

Do not change route planner decisions in this patch. This is only materialization wiring.

### Acceptance criteria

- Preproc upstream regression passes.
- Existing model stage fragment tests pass.
- YOLOv8 graph/model composition tests still pass.

---

## Chapter 14 — Fix BoxDecode adapter-only metadata

> **CUT / DEFERRED (2026-06-04 review).** The early-return at `Model.cpp:4229-4238` (`boxdecode_selected && graph_family != Preproc → nullopt`) does provably shadow the `!rp.enabled` adapter branch — that part is a real code fact. **But the shadowed branch is only reachable when `family == Disabled`** (no quant/tess/resize — a bare-tensor passthrough into BoxDecode; `InputPlanner.cpp:798` default), and no real compiled model exhibiting that topology has been shown. This fixes possibly-dead code, its only test (Ch10) cannot reach it, and it is **orthogonal to the live BoxDecode pose/seg decode failures** (see Chapter 1 "Out of scope"). **Do not land** until someone exhibits a real archive that produces `family == Disabled + boxdecode_selected` and a failing observable. The original fix sketch below is retained for reference.

### File

- `src/model/Model.cpp`

### Current code

```cpp
if (plan.session_route_plan.boxdecode_selected &&
    rp.graph_family != PreprocessGraphFamily::Preproc) {
  if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
    ...
  }
  return std::nullopt;
}
```

### Surgical fix

Only suppress metadata for enabled non-Preproc graph families, not adapter-only routes:

```cpp
if (plan.session_route_plan.boxdecode_selected && rp.enabled &&
    rp.graph_family != PreprocessGraphFamily::Preproc) {
  if (env_bool("SIMA_TYPED_ADAPTER_DEBUG", false)) {
    std::fprintf(stderr,
                 "[typed-adapter] preprocess-meta disabled for non-preproc boxdecode "
                 "graph_family=%s\n",
                 preprocess_graph_family_name(rp.graph_family).c_str());
  }
  return std::nullopt;
}
```

Now this existing branch becomes reachable:

```cpp
if (!rp.enabled) {
  // Adapter-only ingress (quant/tess/quanttess) may still feed boxdecode and
  // therefore must carry preprocess runtime metadata even without a preproc graph.
  ...
  return meta;
}
```

### Acceptance criteria

- Adapter-only BoxDecode metadata regression passes.
- No metadata is added for explicitly unsupported/non-Preproc enabled graph families.

---

## Chapter 15 — Fix archive extraction destination collisions

> **Review revisions (2026-06-04).** Real data-loss bug (verified: `extract_destination_for()` keys on `filename()` only, so `a/config.json` and `b/config.json` overwrite). **But land in two steps to avoid rejecting archives users currently rely on (Risk 4):**
> 1. First ship the collision detection as a **loud warning/diagnostic** (log both colliding source paths), not a hard rejection. Resolve the extracted-directory bypass inconsistency at the same time.
> 2. Promote to a **hard rejection only after an archive-in-the-wild audit** confirms no shipped/vendor archive depends on basename flattening.

### File

- `src/model/ModelArchiveLoader.cpp`

### Current code

```cpp
fs::path extract_destination_for(const fs::path& package_root, const TarEntry& entry) {
  const std::string ext = fs::path(entry.normalized_path).extension().string();
  const fs::path name = fs::path(entry.normalized_path).filename();

  if (entry.entry_class == EntryClass::Json) {
    return package_root / "etc" / name;
  }
  if (ext == ".so") {
    return package_root / "lib" / name;
  }
  if (ext == ".elf") {
    return package_root / "share" / name;
  }

  return package_root / "etc" / name;
}
```

### Implementation detail

Split relative destination from absolute destination:

```cpp
fs::path extract_relative_destination_for(const TarEntry& entry) {
  const std::string ext = fs::path(entry.normalized_path).extension().string();
  const fs::path name = fs::path(entry.normalized_path).filename();

  if (entry.entry_class == EntryClass::Json) {
    return fs::path("etc") / name;
  }
  if (ext == ".so") {
    return fs::path("lib") / name;
  }
  if (ext == ".elf") {
    return fs::path("share") / name;
  }
  return fs::path("etc") / name;
}

fs::path extract_destination_for(const fs::path& package_root, const TarEntry& entry) {
  return package_root / extract_relative_destination_for(entry);
}
```

In `validate_archive(...)`, add:

```cpp
std::unordered_map<std::string, std::string> seen_extract_paths;
```

Inside the entry loop, after `entry` is parsed and duplicate normalized path is checked:

```cpp
if (entry.type == '-' && entry.entry_class != EntryClass::Directory) {
  const std::string rel_dst = extract_relative_destination_for(entry).generic_string();
  auto [it, inserted] = seen_extract_paths.emplace(rel_dst, entry.normalized_path);
  if (!inserted) {
    throw_archive(ModelArchiveErrorClass::InvalidArchive,
                  "invalid_archive: extraction destination collision: '" +
                      entry.normalized_path + "' and '" + it->second +
                      "' both map to '" + rel_dst + "'");
  }
}
```

### Include requirement

`ModelArchiveLoader.cpp` already uses unordered containers. If `unordered_map` is not included, add:

```cpp
#include <unordered_map>
```

### Acceptance criteria

- Collision fixture fails validation before extraction.
- Existing valid archives still extract deterministically.
- Security archive tests pass.

---

# PR 4 — Model `NeatError` normalization

> **Review revisions (2026-06-04).** The contract violation is real (header `Model.h:343-350` promises `NeatError`; impl throws raw `std::runtime_error`/`std::invalid_argument`; `ModelPack.cpp:1108-1110` erases `ModelArchiveError`). Keep this PR, with two corrections:
> - **Scope is Model-only**, via the internal helper refactor (Ch18) — *not* "wrap every public boundary". Delete maximalist framing wherever it appears (esp. Ch21).
> - **Prerequisite caller audit (blocking):** grep for callers catching the exact pre-existing types before flipping them. `NeatError` derives from `std::runtime_error` so `runtime_error` catchers keep working, but `std::invalid_argument` catchers will break — there are 5+ such throw sites (`Model.cpp:1751,1754,2045,2121`, plus `4216`). List and reconcile them in the PR body.
> - Document the known `pyneat.Model`-vs-`Graph` `NeatError` inconsistency rather than silently widening scope to Graph.

---

## Chapter 16 — Add Model-specific error codes

### File

- `include/pipeline/ErrorCodes.h`

### Add constants

Under I/O classes:

```cpp
/// Model archive validation/loading failed.
inline constexpr const char* kModelArchive = "io.model_archive";
/// Model archive extraction/storage failed.
inline constexpr const char* kModelExtract = "io.model_extract";
/// Model metadata JSON is malformed or unreadable.
inline constexpr const char* kModelMetadata = "io.model_metadata";
```

Under misconfiguration classes:

```cpp
/// Invalid Model::Options or unsupported option combination.
inline constexpr const char* kModelOptions = "misconfig.model_options";
/// User input passed to Model::build/run does not match the model ingress contract.
inline constexpr const char* kModelInput = "misconfig.model_input";
/// Model route planner could not select a valid route.
inline constexpr const char* kModelRoute = "misconfig.model_route";
/// Graph composition around a Model fragment is invalid.
inline constexpr const char* kGraphCompose = "misconfig.graph_compose";
/// Python binding received invalid Model/Graph input shape/type.
inline constexpr const char* kPythonInput = "misconfig.python_input";
```

Under build classes:

```cpp
/// Model route materialization failed while creating Nodes/stage contracts.
inline constexpr const char* kModelMaterialize = "build.model_materialize";
```

### Acceptance criteria

- Python module exports new constants if public Python error-code constants are expected.
- Existing code compiles.

---

## Chapter 17 — Preserve `ModelArchiveError` class until public boundary

### File

- `src/model/ModelPack.cpp`

### Current code

```cpp
} catch (const simaai::neat::internal::ModelArchiveError& e) {
  throw std::runtime_error(std::string("ModelPack: ") + e.what());
}
```

### Change

```cpp
} catch (const simaai::neat::internal::ModelArchiveError&) {
  throw;
}
```

### Reason

`ModelArchiveError::code()` is useful at the public boundary. Converting to `std::runtime_error` too early loses:

- invalid archive
- path traversal
- schema error
- unsupported version
- size limit
- unsupported extension
- output storage unavailable

### Acceptance criteria

- Public `Model` constructor can catch `internal::ModelArchiveError`.
- Archive unit tests still pass.

---

## Chapter 18 — Add Model error helper

### File

- `src/model/Model.cpp`

### Includes

Add if not already available:

```cpp
#include "pipeline/ErrorCodes.h"
#include "pipeline/NeatError.h"

#define SIMA_NEAT_INTERNAL
#include "pipeline/internal/ErrorUtil.h"
#undef SIMA_NEAT_INTERNAL
```

### Helper implementation

Place in anonymous namespace near other Model helpers:

```cpp
[[noreturn]] void throw_model_error(std::string_view code,
                                    std::string_view where,
                                    std::string_view detail,
                                    std::string_view hint = {}) {
  std::string summary;
  summary.reserve(where.size() + detail.size() + 2U);
  summary.append(where.begin(), where.end());
  summary.append(": ");
  summary.append(detail.begin(), detail.end());

  auto rep = pipeline_internal::error_util::make_report(code, summary, {}, hint);
  throw NeatError(pipeline_internal::error_util::decorate_error(rep.error_code, rep.repro_note),
                  std::move(rep));
}

const char* code_for_archive_error(internal::ModelArchiveErrorClass code) {
  using C = internal::ModelArchiveErrorClass;
  switch (code) {
  case C::OutputStorageUnavailable:
    return error_codes::kModelExtract;
  case C::InvalidArchive:
  case C::PathTraversal:
  case C::SchemaError:
  case C::UnsupportedVersion:
  case C::SizeLimitExceeded:
  case C::UnsupportedExtension:
  default:
    return error_codes::kModelArchive;
  }
}
```

### Report shape rule

Do not invent new `GraphReport` fields. Use:

- `error_code`
- `repro_note`
- optionally `pipeline_string` if a pipeline exists

Early setup failures usually have no GStreamer pipeline yet.

---

## Chapter 19 — Wrap `Model` constructor

### File

- `src/model/Model.cpp`

### Current code

```cpp
Model::Model(const std::string& model_path, const Options& opt) {
  impl_ = std::make_unique<Impl>(model_path, opt);
}
```

### Replacement

```cpp
Model::Model(const std::string& model_path, const Options& opt) {
  try {
    impl_ = std::make_unique<Impl>(model_path, opt);
  } catch (const NeatError&) {
    throw;
  } catch (const internal::ModelArchiveError& e) {
    throw_model_error(code_for_archive_error(e.code()),
                      "Model::Model",
                      e.what(),
                      "Verify the .tar.gz path, archive contents, manifest, and MPK contract.");
  } catch (const nlohmann::json::exception& e) {
    throw_model_error(error_codes::kModelMetadata,
                      "Model::Model",
                      e.what(),
                      "Verify JSON files inside the model archive.");
  } catch (const std::invalid_argument& e) {
    throw_model_error(error_codes::kModelOptions, "Model::Model", e.what());
  } catch (const std::exception& e) {
    throw_model_error(error_codes::kModelRoute, "Model::Model", e.what());
  }
}
```

### Acceptance criteria

- Missing/bad archives throw `NeatError`.
- Archive category is not lost.
- Existing callers catching `std::exception` still work because `NeatError` derives from `std::runtime_error`.

---

## Chapter 20 — Wrap `Model::metadata()`

### File

- `src/model/Model.cpp`

### Current code

```cpp
nlohmann::json j;
in >> j;
if (!j.is_object())
  return out;
```

JSON exceptions escape raw.

### Replacement pattern

```cpp
std::unordered_map<std::string, std::string> Model::metadata() const {
  try {
    std::unordered_map<std::string, std::string> out;
    const std::string path = impl_->pack.etc_dir() + "/metadata.json";
    std::ifstream in(path);
    if (!in.is_open())
      return out;
    nlohmann::json j;
    in >> j;
    if (!j.is_object())
      return out;
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (it.value().is_string()) {
        out[it.key()] = it.value().get<std::string>();
      } else {
        out[it.key()] = it.value().dump();
      }
    }
    return out;
  } catch (const NeatError&) {
    throw;
  } catch (const nlohmann::json::exception& e) {
    throw_model_error(error_codes::kModelMetadata,
                      "Model::metadata",
                      e.what(),
                      "metadata.json must be valid JSON object data.");
  } catch (const std::exception& e) {
    throw_model_error(error_codes::kModelMetadata, "Model::metadata", e.what());
  }
}
```

### Acceptance criteria

- Missing metadata still returns empty map.
- Malformed metadata throws `NeatError` with `io.model_metadata`.

---

## Chapter 21 — Wrap public `Model::build/run` boundaries

### File

- `src/model/Model.cpp`

### Principle

Do not wrap every internal helper. Wrap public Model methods so users get consistent errors.

### Minimal helper

```cpp
template <typename Fn>
auto model_boundary(std::string_view where, std::string_view fallback_code, Fn&& fn)
    -> decltype(fn()) {
  try {
    return fn();
  } catch (const NeatError&) {
    throw;
  } catch (const std::invalid_argument& e) {
    throw_model_error(fallback_code, where, e.what());
  } catch (const std::runtime_error& e) {
    throw_model_error(fallback_code, where, e.what());
  } catch (const std::exception& e) {
    throw_model_error(fallback_code, where, e.what());
  }
}
```

### Implementation style

Avoid giant nested lambdas in huge functions if readability suffers. Preferred approach:

1. Rename current body to an internal helper:

```cpp
Model::Runner Model::build_tensor_list_impl(const TensorList& inputs,
                                            const RouteOptions& opt,
                                            const RunOptions& run_opt) {
  ... old body ...
}
```

2. Public function becomes:

```cpp
Model::Runner Model::build(const neat::TensorList& inputs,
                           const Model::RouteOptions& opt,
                           const neat::RunOptions& run_opt) {
  return model_boundary("Model::build(TensorList)", error_codes::kModelInput, [&] {
    return build_tensor_list_impl(inputs, opt, run_opt);
  });
}
```

Because member private declarations may be cumbersome, these helpers can be file-local free functions that take `Model&` only if they have access through existing public/internal access. If not, use lambda wrapping directly for the first patch.

### Methods to wrap first

- `Model::build()`
- `Model::build(const RouteOptions&)`
- `Model::build(const RunOptions&)`
- `Model::build(const RouteOptions&, const RunOptions&)`
- `Model::build(const TensorList&, ...)`
- `Model::build(const Sample&, ...)`
- OpenCV `Model::build(const std::vector<cv::Mat>&, ...)`
- `Model::run(const TensorList&, int)`
- `Model::run(const Sample&, int)`
- OpenCV `Model::run(const std::vector<cv::Mat>&, int)`

### Suggested fallback codes

- Empty input / wrong ingress count / wrong image mode: `error_codes::kModelInput`
- Route materialization failures while building nodes: `error_codes::kModelMaterialize`
- Graph build failures that are already `NeatError`: preserve.
- Generic route planning not caught earlier: `error_codes::kModelRoute`

### Acceptance criteria

- Existing runtime `NeatError`s are not double-wrapped.
- Public input misuse gets `misconfig.model_input`.
- Build/materialization failures include method name in `repro_note`.

---

# PR 4B — Python Model input error parity

---

## Chapter 22 — Convert Python Model input misuse to `pyneat.NeatError`

### File

- `python/src/module.cpp`

### Current behavior

Python helpers throw `std::runtime_error`:

```cpp
throw std::runtime_error("expected list/tuple of Tensor or DLPack-compatible inputs");
```

`reject_single_tensor_or_sample(...)` also throws raw runtime errors.

### Implementation detail

Add helper near `format_neat_error_message(...)`:

```cpp
[[noreturn]] void throw_python_input_error(const char* where, const std::string& detail) {
  GraphReport rep;
  rep.error_code = error_codes::kPythonInput;
  rep.repro_note = std::string(where ? where : "pyneat") + ": " + detail;
  throw NeatError("[" + rep.error_code + "] " + rep.repro_note, std::move(rep));
}
```

Change `reject_single_tensor_or_sample(...)`:

```cpp
void reject_single_tensor_or_sample(const nb::object& input, const char* where) {
  if (nb::isinstance<Sample>(input)) {
    throw_python_input_error(where, "pass [sample] instead of a single Sample");
  }
  if (nb::isinstance<Tensor>(input)) {
    throw_python_input_error(where, "pass [tensor] instead of a single Tensor");
  }
}
```

For conversion helpers, there are two options.

### Option A — narrow Model-only wrapping

Leave low-level helpers raw, and wrap Model binding lambdas:

```cpp
.def("run", [](neat::Model& model, nb::object input, int timeout_ms, bool copy) -> nb::object {
  try {
    reject_single_tensor_or_sample(input, "Model.run");
    ...
  } catch (const NeatError&) {
    throw;
  } catch (const std::exception& e) {
    throw_python_input_error("Model.run", e.what());
  }
})
```

### Option B — globally convert helper errors

Change helper throws directly:

```cpp
throw_python_input_error("pyneat input conversion",
                         "expected list/tuple of Tensor or DLPack-compatible inputs");
```

Recommendation: Option A first, because it limits behavior change to Model-facing APIs.

### Python tests

In `python/tests/test_api_surface.py` or a new model input error test:

```python
def test_model_rejects_single_tensor_with_neat_error(model_fixture, tensor):
    model = pyneat.Model(model_fixture)
    with pytest.raises(pyneat.NeatError) as ei:
        model.run(tensor)
    assert ei.value.error_code == pyneat.ERROR_PYTHON_INPUT
    assert "pass [tensor]" in str(ei.value)
```

### Acceptance criteria

- Python single Tensor/Sample misuse throws `pyneat.NeatError`.
- `.error_code == "misconfig.python_input"`.
- Existing list/tuple usage still works.

---

# PR 5 — Python parity bindings

---

## Chapter 23 — Bind stable Model introspection APIs

### File

- `python/src/module.cpp`

### Current gap

C++ has:

- `input_specs()`
- `output_specs()`
- `compiled_batch_size()`
- `resolved_preprocess_plan()`
- `preprocess_requirements()`
- `info()`
- `input_appsrc_options_list()`

Python currently exposes only:

- `input_spec()`
- `output_spec()`
- `metadata()`
- basic graph/build/run methods

### Implementation detail — bind TensorSpec/TensorConstraint if not bound

Search first for existing `TensorConstraint` binding. If none exists, add:

```cpp
nb::class_<simaai::neat::TensorSpec>(m, "TensorSpec")
    .def_ro("rank", &simaai::neat::TensorSpec::rank)
    .def_ro("shape", &simaai::neat::TensorSpec::shape)
    .def_ro("dtypes", &simaai::neat::TensorSpec::dtypes);
```

If `dtypes` cannot be bound directly because of enum-vector conversion, expose a summary:

```cpp
.def("to_dict", [](const simaai::neat::TensorSpec& spec) {
  nb::dict d;
  d["rank"] = spec.rank;
  d["shape"] = spec.shape;
  nb::list dtypes;
  for (const auto dtype : spec.dtypes) {
    dtypes.append(nb::cast(dtype));
  }
  d["dtypes"] = dtypes;
  return d;
})
```

### Implementation detail — add Model bindings

In the `nb::class_<simaai::neat::Model>` chain:

```cpp
.def("input_specs", &simaai::neat::Model::input_specs)
.def("output_specs", &simaai::neat::Model::output_specs)
.def("compiled_batch_size", &simaai::neat::Model::compiled_batch_size)
.def("input_appsrc_options_list", &simaai::neat::Model::input_appsrc_options_list,
     "tensor_mode"_a)
```

For complex APIs, start with dict summaries rather than full deep class binding.

### `info_dict()` binding

```cpp
.def("info_dict", [](const simaai::neat::Model& model) {
  const auto info = model.info();
  nb::dict d;
  d["mpk_json_path"] = info.mpk_json_path;
  d["model_name"] = info.model_name;

  nb::dict needs;
  needs["pre_quantization"] = info.needs.pre_quantization;
  needs["pre_tessellation"] = info.needs.pre_tessellation;
  needs["pre_cast"] = info.needs.pre_cast;
  needs["post_detessellation"] = info.needs.post_detessellation;
  needs["post_dequantization"] = info.needs.post_dequantization;
  needs["post_cast"] = info.needs.post_cast;
  d["needs"] = needs;

  nb::dict capabilities;
  capabilities["has_pre_quantization"] = info.capabilities.has_pre_quantization;
  capabilities["has_pre_tessellation"] = info.capabilities.has_pre_tessellation;
  capabilities["has_pre_cast"] = info.capabilities.has_pre_cast;
  capabilities["has_post_detessellation"] = info.capabilities.has_post_detessellation;
  capabilities["has_post_dequantization"] = info.capabilities.has_post_dequantization;
  capabilities["has_post_cast"] = info.capabilities.has_post_cast;
  capabilities["has_post_boxdecode"] = info.capabilities.has_post_boxdecode;
  d["capabilities"] = capabilities;

  nb::dict selection;
  selection["include_preprocess_stage"] = info.selection.include_preprocess_stage;
  selection["include_postprocess_stage"] = info.selection.include_postprocess_stage;
  selection["infer_only"] = info.selection.infer_only;
  selection["preprocess_graph"] = info.selection.preprocess_graph;
  selection["selected_post_kind"] = info.selection.selected_post_kind;
  d["selection"] = selection;

  nb::dict topology;
  topology["physical_outputs"] = info.output_topology.physical_outputs;
  topology["logical_outputs"] = info.output_topology.logical_outputs;
  topology["packed_outputs"] = info.output_topology.packed_outputs;
  d["output_topology"] = topology;

  d["pre_kernels"] = info.pre_kernels;
  d["post_kernels"] = info.post_kernels;
  d["warnings"] = info.warnings;
  return d;
})
```

### `preprocess_requirements_dict()` binding

```cpp
.def("preprocess_requirements_dict", [](const simaai::neat::Model& model) {
  const auto req = model.preprocess_requirements();
  nb::dict d;
  d["has_preproc_stage"] = req.has_preproc_stage;
  d["quant_needed"] = req.quant_needed;
  d["tess_needed"] = req.tess_needed;
  d["input_media_type"] = req.input_media_type;
  d["input_format"] = req.input_format;
  d["output_format"] = req.output_format;
  d["output_dtype"] = req.output_dtype;
  d["output_shape"] = req.output_shape;
  d["axis_perm"] = req.axis_perm;
  d["slice_shape"] = req.slice_shape;
  d["q_scale"] = req.q_scale;
  d["q_zp"] = req.q_zp;
  return d;
})
```

### Acceptance criteria

- Python can ask the same basic questions as C++.
- Complex returned values are stable dicts, not half-bound fragile C++ internals.

---

## Chapter 24 — Bind missing `ModelRouteOptions` fields

### File

- `python/src/module.cpp`

### Current binding

```cpp
nb::class_<simaai::neat::Model::RouteOptions>(m, "ModelRouteOptions")
    .def(nb::init<>())
    .def_rw("include_input", ...)
    .def_rw("include_output", ...)
    .def_rw("expose_all_outputs", ...)
    .def_rw("upstream_name", ...)
    .def_rw("name_suffix", ...)
    .def_rw("buffer_name", ...);
```

Missing fields:

- `verbose`
- `processcvu_requested_run_target`
- `processcvu`
- `processmla`
- `prepared_runner`
- `async_queue_depth`

### Add

```cpp
.def_rw("verbose", &simaai::neat::Model::RouteOptions::verbose)
.def_rw("processcvu_requested_run_target",
        &simaai::neat::Model::RouteOptions::processcvu_requested_run_target)
.def_rw("processcvu", &simaai::neat::Model::RouteOptions::processcvu)
.def_rw("processmla", &simaai::neat::Model::RouteOptions::processmla)
.def_rw("prepared_runner", &simaai::neat::Model::RouteOptions::prepared_runner)
.def_rw("async_queue_depth", &simaai::neat::Model::RouteOptions::async_queue_depth);
```

### Acceptance criteria

- `python/tests/test_api_surface.py` asserts these attributes exist.

---

## Chapter 25 — Bind `ModelRunner` diagnostics/report APIs

### File

- `python/src/module.cpp`

### Current binding ends at

```cpp
.def("close", &simaai::neat::Model::Runner::close);
```

### Add

```cpp
.def("close", &simaai::neat::Model::Runner::close)
.def("close_input", &simaai::neat::Model::Runner::close_input)
.def("stats", &simaai::neat::Model::Runner::stats)
.def("measurement_summary", &simaai::neat::Model::Runner::measurement_summary)
.def("diag_snapshot", &simaai::neat::Model::Runner::diag_snapshot)
.def("report", &simaai::neat::Model::Runner::report,
     "options"_a = simaai::neat::RunReportOptions{})
.def("metrics_report",
     static_cast<std::string (simaai::neat::Model::Runner::*)(
         const simaai::neat::RuntimeMetricsOptions&,
         simaai::neat::RuntimeMetricsFormat) const>(
         &simaai::neat::Model::Runner::metrics_report),
     "options"_a = simaai::neat::RuntimeMetricsOptions{},
     "format"_a = simaai::neat::RuntimeMetricsFormat::Text);
```

If `RunStats`, `RunMeasurementSummary`, or diagnostic structs are not bound well enough for direct return, start with string-only `report()` and `metrics_report()`.

### Acceptance criteria

- Python `ModelRunner` has the same diagnostics entry points as `Run`, or intentionally documented subset.

---

# Lower-priority cleanup

---

## Chapter 26 — Make `pack_for_sync()` thread-safety explicit

### File

- `src/model/Model.cpp`

### Current code

```cpp
mutable std::optional<internal::ModelPack> sync_pack;

const internal::ModelPack& pack_for_sync() const {
  if (!sync_pack.has_value()) {
    sync_pack = pack.clone_with_buffers(1, 1);
  }
  return *sync_pack;
}
```

This is not explicitly thread-safe if accessed from multiple threads before initialization.

### Option A — code fix

```cpp
mutable std::once_flag sync_pack_once;
mutable std::unique_ptr<internal::ModelPack> sync_pack;

const internal::ModelPack& pack_for_sync() const {
  std::call_once(sync_pack_once, [&] {
    sync_pack = std::make_unique<internal::ModelPack>(pack.clone_with_buffers(1, 1));
  });
  return *sync_pack;
}
```

Need include:

```cpp
#include <mutex>
#include <memory>
```

Already likely present.

### Option B — documentation only

If current call graph always serializes via `sync_mu`, document that invariant in a comment.

Recommendation: code fix is small and robust, but do it after the higher-value fixes.

---

## Chapter 27 — Model output semantic introspection

> **CUT (2026-06-04 review).** Speculative new public API surface (`ModelOutputSemanticSpec`, `semantic_output_specs()`, Detection/Segmentation/Pose kinds) that the plan itself defers to "later". It does not fix a verified bug and inflates the document. Drop it from this plan; if the need is real, propose it as its own design RFC. Sketch retained below for reference only.

### Current issue

`Model::output_specs()` for BoxDecode currently returns a generic UInt8/rank unknown style spec:

```cpp
if (has_box) {
  spec.dtypes = {TensorDType::UInt8};
  spec.rank = -1;
  push_spec(std::move(spec));
  return specs;
}
```

This loses semantic information: boxes/classes/scores layout, serialized detection format, etc.

### Plan

Do not solve this in the first wave. Later, add a semantic output descriptor:

```cpp
struct ModelOutputSemanticSpec {
  enum class Kind { Tensor, DetectionBoxes, SegmentationMask, PoseKeypoints };
  Kind kind = Kind::Tensor;
  std::string encoding;
  int max_detections = 0;
  int num_classes = 0;
  std::vector<std::string> fields;
};
```

Expose through:

```cpp
std::vector<ModelOutputSemanticSpec> semantic_output_specs() const;
```

This should be done together with YOLO detection/segmentation/pose unification work, not as a drive-by.

---

# Validation

---

## Chapter 28 — Local checks after each PR

From `/home/docker/sima-cli/core_graph_changes`:

```bash
git diff --check
```

Focused build:

```bash
cmake --build build-codex-graph-sdk --target \
  unit_model_stage_fragments_test \
  unit_model_route_materialization_test \
  unit_model_archive_loader_test \
  unit_model_error_contract_test \
  unit_boxdecode_render_manifest_from_model_test \
  unit_model_metadata_test \
  -j$(nproc)
```

Before executing built binaries, check architecture:

```bash
file build-codex-graph-sdk/tests/unit_model_stage_fragments_test
```

If ARM64/aarch64, run on DevKit, not locally.

Preferred if available:

```bash
dk /home/docker/sima-cli/tmp/devkit_env_exec.sh \
  /workspace/core_graph_changes/build-codex-graph-sdk/tests/unit_model_stage_fragments_test
```

Fallback direct SSH only if `dk` unavailable.

---

## Chapter 29 — Python checks

Run API surface tests:

```bash
pytest -q python/tests/test_api_surface.py
```

If tests need installed pyneat, run in the same environment CI uses. Add tests for:

```python
def test_model_route_options_exposes_new_fields():
    opt = pyneat.ModelRouteOptions()
    for name in [
        "verbose",
        "processcvu_requested_run_target",
        "processcvu",
        "processmla",
        "prepared_runner",
        "async_queue_depth",
    ]:
        assert hasattr(opt, name), name
```

And:

```python
def test_model_public_old_names_hidden():
    assert not hasattr(pyneat.Model, "build_tensors")
    assert not hasattr(pyneat.Model, "run_tensors")
```

This already exists but keep it passing.

---

## Chapter 30 — Broader CI-style validation

After focused tests pass:

```bash
ctest --test-dir build-codex-graph-sdk --output-on-failure
```

If the repo has a faster CI wrapper, prefer the same command CI uses. Do not mix installed runtime artifacts with build-tree plugins unless the test explicitly expects that; previous YOLO/preproc tests showed runtime/plugin ABI mixing can create misleading shutdown errors.

For model/graph changes, prioritize:

- `unit_model_stage_fragments_test`
- `unit_model_graph_seed_preproc_specialization_test`
- `unit_model_route_planner_test`
- `unit_model_input_spec_contract_test`
- `unit_model_output_spec_contract_test`
- `unit_boxdecode_render_manifest_from_model_test`
- `graph_migration_unified_yolov8_graph_model_composition_test`
- `hybrid_graph_stage_model_test`
- `model_resnet50_multi_test`

---

# Risk register

---

## Chapter 31 — Risks and mitigations

### Risk 1 — Preproc upstream fix changes rendered pipeline names

Mitigation:

- Keep stage names unchanged.
- Only change upstream wiring.
- Add regression tests on route contents.

### Risk 2 — `NeatError` conversion breaks tests expecting exact exception type

Mitigation:

- Public API should catch `std::exception` or `NeatError`, not exact `std::invalid_argument`.
- Internal helper tests can still expect raw exceptions if they call internals directly.
- Update only public-boundary tests.

### Risk 3 — Python dict summaries become accidental API forever

Mitigation:

- Name them explicitly `info_dict()` / `preprocess_requirements_dict()`.
- Document that they are stable summaries, not exact C++ object mirrors.

### Risk 4 — Archive collision validation rejects archives users currently rely on

Mitigation:

- This rejection is correct; existing behavior overwrites files silently.
- Error message must be clear and actionable.

### Risk 5 — Model metadata for adapter-only BoxDecode may be too broad

> **Superseded by the 2026-06-04 review: Ch14 is CUT.** The risk is moot until Ch14 is revived with a real failing model. If revived, the mitigation below stands. Note this is orthogonal to the live BoxDecode pose/seg decode failures (Chapter 1 "Out of scope").

Mitigation (only if Ch14 is revived):

- Condition should be exactly `boxdecode_selected && !rp.enabled` for adapter-only route.
- Keep enabled non-Preproc graph-family suppression.
- Add a test that actually drives `build_pipeline_nodes` with a `family == Disabled + boxdecode_selected` model (the Ch10 draft does not — it is cut).

---

# Final checklist

---

## Chapter 32 — Definition of done

Docs:

- [ ] `Model.h` examples use `namespace neat = simaai::neat;`.
- [ ] `ModelGroups.h` says `Infer()` is inference-stage only.
- [ ] Main docs prefer `Graph::add(model)`.
- [ ] Tutorial 001 Python uses `outputs[0]`, not `sample.tensor`, for tensor input.
- [ ] Tutorial 013 explicitly says `graph.add(model)` is full route.

Tests:

- [ ] Preproc self-upstream regression exists.
- [ ] **Multi-region preproc chaining regression exists** (proves region N+1 chains off region N — required before Ch13 lands).
- [ ] Archive extraction collision regression exists.
- [ ] Model public error contract regression exists.
- [ ] Python API surface tests cover new bindings.
- [ ] ~~Adapter-only BoxDecode metadata regression exists.~~ **(CUT — Ch10, dead test.)**

Fixes:

- [ ] Pre-regions chain through previous pre-region output (all regions, not just self-link).
- [ ] MLA consumes final pre-stage, not external decoder when pre-stage exists.
- [ ] Archive flattened destination collisions are **warned** (hard-rejected only after archive audit).
- [ ] Public Model setup/build/run failures map to `NeatError` (after `std::invalid_argument` caller audit).
- [ ] Python Model input misuse maps to `pyneat.NeatError`.
- [ ] ~~Adapter-only BoxDecode metadata branch is reachable.~~ **(CUT/DEFERRED — Ch14, unproven reachability.)**

Explicitly NOT in this plan's Definition of Done:

- [ ] The live BoxDecode **pose/seg decode** fix (`YoloV26Pose` enum/token, `compute_pose` dtype/overflow, `num_classes` gate, physical-vs-logical channel depth) — **separate ticket** (Chapter 1 "Out of scope").

Validation:

- [ ] `git diff --check` clean.
- [ ] Focused unit tests pass.
- [ ] Python API tests pass.
- [ ] Broader CI-style tests pass or failures are unrelated and documented.
