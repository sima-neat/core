# Model Fix Plan

Investigation date: 2026-05-30
Last updated: 2026-05-31
Scope: `Model` API, implementation, Python bindings, docs, tutorials, and related grouped nodes in `core_graph_changes`.

This file is a working document. It captures what the subagent investigation found, what looks broken or risky, and a proposed way for us to solve it together without randomly patching symptoms.

---

## -1. Phase -1: NeatError investigation

Decision from 2026-05-31: do **not** treat the current hybrid error behavior as final. Model-facing
setup/planning/composition failures should become real `NeatError`s with useful `GraphReport`
payloads before docs promise that contract broadly.

### Phase -1 result, 2026-05-31

The investigation confirms the header-level contract is ahead of implementation:

- `include/pipeline/NeatError.h:4-10` says `NeatError` is the framework's exception type and that
  `Graph::build()`, `Model::run()`, `Run::push()`, etc. throw it on framework failures.
- `include/model/Model.h:343-350` and `include/model/Model.h:539-585` already document
  constructor/build/run failures as `NeatError`.
- Runtime/build code mostly follows that contract once a pipeline/session exists.
- Model construction, route planning, model materialization, graph composition around models, and
  Python binding input conversion still throw raw `std::runtime_error`, `std::invalid_argument`,
  JSON exceptions, OpenCV exceptions, or archive-loader exceptions erased into `std::runtime_error`.

So Phase -1 is not about inventing a new concept; it is about making implementation match public
contract.

### Review corrections incorporated before coding

The independent review is accepted. These changes are now part of the plan before implementation:

1. **Python parity is decided, not left open.** For Model-facing Python APIs we choose full parity:
   binding-level input misuse becomes `pyneat.NeatError` with `misconfig.python_input`. List/tuple
   remains required.
2. **Python plumbing is explicit.** Add `kPythonInput` / `ERROR_PYTHON_INPUT`, then throw a real C++
   `NeatError` from the binding boundary so the existing nanobind translator creates
   `pyneat.NeatError`.
3. **`metadata()` needs explicit catch ordering.** Catch `nlohmann::json::exception` before generic
   method-level mappings so malformed metadata becomes `io.model_metadata`, not a fallback code.
4. **OpenCV overloads need explicit `cv::Exception` handling.** Under `SIMA_WITH_OPENCV`, catch
   `const cv::Exception&` in Model cv::Mat paths and map to `misconfig.model_input`.
5. **Helper shape must match `GraphReport`.** `GraphReport` is a plain aggregate and has no
   `where`, `detail`, `model_path`, `hint`, or `report_json` fields. Helpers must default-construct
   it, assign `error_code`, and fold all context into `repro_note`. Python `.report_json` is
   synthesized by the translator from `GraphReport::to_json()`.
6. **`model.graph()` does not throw `NeatError` today.** The `catch (const NeatError&) { throw; }`
   guard is still required, but it becomes load-bearing after Phase -1 adds wrapped Model paths and
   for downstream build/runtime calls that already throw `NeatError`.
7. **Do not overclaim per-throw granularity from method-level wrapping.** A single method wrapper
   cannot know which of many internal throws fired. Phase -1 will use targeted conversion at known
   user-facing validation/call sites plus coarse fallback wrappers only where unavoidable.
8. **Caller compatibility needs an audit/test.** Public Model errors changing from
   `std::invalid_argument` to `NeatError` is acceptable because `NeatError` derives from
   `std::runtime_error`, but callers/tests expecting exact `std::invalid_argument` at public Model
   boundaries must be updated or preserved intentionally. Internal helper tests can still catch
   `std::invalid_argument` because we are not changing every internal helper.

### Existing diagnostics infrastructure

Verified pieces that already exist and should be reused:

- `NeatError` carries a `GraphReport`; `GraphReport` already has the fields we need for early setup
  failures: `error_code`, `repro_note`, `pipeline_string`, `repro_gst_launch`, `repro_env`, and
  `to_json()`.
- `pipeline/internal/ErrorUtil` already provides report/decorating helpers:
  - `decorate_error(...)`
  - `append_hint(...)`
  - `make_report(...)`
  - `throw_session_error(...)`
- Existing canonical codes in `include/pipeline/ErrorCodes.h` are currently sparse:
  - `misconfig.pipeline_shape`
  - `misconfig.caps`
  - `misconfig.input_shape`
  - `misconfig.runtime_abi_mismatch`
  - `build.parse_launch`
  - `runtime.pull`
  - `io.parse`
  - `io.open`
  - `infra.dispatcher_unavailable`
- Python already has a real `pyneat.NeatError` binding:
  - `python/src/module.cpp:1122-1148` registers a translator for C++ `NeatError`.
  - Python exception attrs currently include `.error_code`, `.repro_note`, `.pipeline_string`, and
    `.report_json`.
  - `.report_json` is **not** a `GraphReport` field; it is synthesized by the translator using
    `GraphReport::to_json()`.
  - `GraphReport` itself is also bound at `python/src/module.cpp:1500-1514`.

Conclusion: we do **not** need a new Python exception class. We need C++ and Python binding
boundaries to throw C++ `NeatError` consistently.

### Confirmed raw-exception gaps

#### 1. `Model` construction and archive handling

- `Model::Model(path, options)` directly constructs `Impl` with no wrapping
  (`src/model/Model.cpp:4093-4097`).
- Archive loader has useful internal categories:
  - `InvalidArchive`
  - `PathTraversal`
  - `SchemaError`
  - `UnsupportedVersion`
  - `SizeLimitExceeded`
  - `UnsupportedExtension`
  - `OutputStorageUnavailable`
- `ModelArchiveError::code()` is queryable at catch time.
- But `ModelPack` currently catches `ModelArchiveError` and rethrows
  `std::runtime_error("ModelPack: ...")` (`src/model/ModelPack.cpp:1108-1110`), losing the enum
  category before public `Model` can map it cleanly.

Purpose of change: preserve/remap archive categories so bad archive, malicious archive, unsupported
archive, and extraction/storage failure are bucketed correctly.

#### 2. Model metadata/introspection

- `Model::metadata()` parses `metadata.json` with `in >> j` and lets JSON parse/type exceptions
  escape raw (`src/model/Model.cpp:6283-6300`).
- Malformed metadata must become `NeatError` with `io.model_metadata`.
- Catch order matters: `catch (const nlohmann::json::exception&)` must run before any generic
  fallback mapping.

Purpose of change: public read-only Model APIs should not leak raw JSON exceptions.

#### 3. Model planning/materialization/build/run

Representative raw exceptions verified:

- Empty or wrong input collections before any `Graph::build(...)` call:
  - `Model::build: empty tensor list` (`src/model/Model.cpp:6639-6644`)
  - image/tensor ingress-count mismatches (`src/model/Model.cpp:6653-6661`)
  - `Model::build: empty sample list` (`src/model/Model.cpp:6695-6700`)
  - `Model::run: empty input list` (`src/model/Model.cpp:6785-6788`)
- Planner and route constraints:
  - bad normalize stddev throws `std::invalid_argument`
    (`src/model/InputPlanner.cpp:226`)
  - unsupported graph family throws `std::invalid_argument`
    (`src/model/InputPlanner.cpp:278`, `src/model/InputPlanner.cpp:323`)
  - missing MLA stage throws `std::runtime_error` (`src/model/InputPlanner.cpp:557`)
  - route hard failures throw `std::runtime_error` (`src/model/RoutePlanner.cpp:3931-3955`)
- Model materialization helpers in `src/model/Model.cpp` have many direct
  `std::runtime_error` / `std::invalid_argument` throws while creating pre/infer/post route nodes.

Purpose of change: wrap only the public Model boundary and targeted user-facing validation points,
not every internal helper.

#### 4. OpenCV `cv::Mat` paths

- OpenCV-guarded Model overloads exist under `SIMA_WITH_OPENCV`.
- `cv::Exception` is not `std::runtime_error` and was missing from the original catch policy.

Purpose of change: malformed/unsupported `cv::Mat` input on Model cv::Mat paths should produce
`NeatError` with `misconfig.model_input`.

#### 5. Graph composition around Model fragments

- `Graph::add(const Model&)` lowers by calling `model.graph(opt)` then `add(fragment)` with no
  wrapping (`src/pipeline/graph/GraphModel.cpp:7-13`).
- Public fragment composition errors still throw raw `std::runtime_error`, e.g. endpoint inference,
  moved-from fragments, ambiguous add after branching, multiple input route processors
  (`src/pipeline/graph/Graph.cpp:991-1048`).
- Model import/connect calls `model.graph()` and then imports the fragment
  (`src/pipeline/graph/Graph.cpp:1311-1330`).
- Runtime graph compilation has model-specific topology errors, such as missing/duplicate/unexpected
  producers for multi-input Model ingress endpoints (`src/pipeline/runtime/ExecutionGraphCompiler.cpp:704-719`).

Purpose of change: `Graph::add(model)` and `Graph::connect(... model ...)` are part of the Model UX;
composition failures involving them should be `NeatError` with `misconfig.graph_compose`.

#### 6. Python binding-level input errors

- The Python binding intentionally requires list/tuple for Model input; that decision stands.
- Binding code currently throws raw `std::runtime_error` for Python API misuse:
  - non-list/tuple input (`python/src/module.cpp:949-975`)
  - single `Tensor` / `Sample` instead of `[tensor]` / `[sample]`
    (`python/src/module.cpp:1048-1057`)
  - sample/tensor list conversion errors (`python/src/module.cpp:1021-1046`)

Purpose of change: for Model/ModelRunner Python APIs only, convert these binding-boundary failures
into C++ `NeatError` with `misconfig.python_input`, then let the existing translator surface
`pyneat.NeatError`.

### Error-code taxonomy to add

Do **not** create deep dotted families like `model.archive.schema_error` yet, because the current
canonical taxonomy is two-level `domain.reason`. Use existing domains plus model-specific reasons:

| Code | Use for |
| --- | --- |
| `io.model_archive` | Invalid/corrupt/malicious/unsupported model archive and archive schema/version/extension/size errors. |
| `io.model_extract` | Unable to extract/write/cache model archive content, including output-storage-unavailable. |
| `io.model_metadata` | Malformed/unparseable `metadata.json`; missing optional metadata remains non-error. |
| `misconfig.model_options` | Bad caller-provided `Model::Options` / `RouteOptions`, bad preprocess intent, invalid thresholds, unsupported placement request. |
| `misconfig.model_input` | Empty inputs, wrong number of ingress inputs, bad `cv::Mat`, wrong tensor/sample collection shape before build. |
| `misconfig.model_route` | Manifest route cannot satisfy the requested SiMa route: missing MLA stage, unsupported graph family, impossible route combination. |
| `build.model_materialize` | Failure while turning a valid route plan into nodes/fragments/configs. |
| `misconfig.graph_compose` | Public Graph composition/topology errors, especially involving `Graph::add(model)` / `Graph::connect(... model ...)`. |
| `misconfig.python_input` | Python Model/ModelRunner binding input misuse before C++ Model execution starts. |

Existing codes should still be used where they fit exactly:

- Missing files/devices: `io.open`.
- Generic JSON/config parse outside model metadata/archive: `io.parse`.
- Pipeline topology shape after graph lowering: `misconfig.pipeline_shape`.
- Runtime pull/push/session failures after a run exists: existing runtime/build codes.

Python exports to add in `python/src/module.cpp`:

```cpp
ERROR_MODEL_ARCHIVE
ERROR_MODEL_EXTRACT
ERROR_MODEL_METADATA
ERROR_MODEL_OPTIONS
ERROR_MODEL_INPUT
ERROR_MODEL_ROUTE
ERROR_MODEL_MATERIALIZE
ERROR_GRAPH_COMPOSE
ERROR_PYTHON_INPUT
```

### Archive-error mapping

Preferred mapping for `ModelArchiveErrorClass`:

| Archive class | Proposed code |
| --- | --- |
| `InvalidArchive` | `io.model_archive` |
| `PathTraversal` | `io.model_archive` |
| `SchemaError` | `io.model_archive` |
| `UnsupportedVersion` | `io.model_archive` |
| `SizeLimitExceeded` | `io.model_archive` |
| `UnsupportedExtension` | `io.model_archive` |
| `OutputStorageUnavailable` | `io.model_extract` |

Implementation note: stop erasing the archive enum in `ModelPack.cpp`, or convert it to `NeatError`
there before the enum is lost.

### Report shape for setup-time errors

For setup/planning/model errors, `GraphReport` should be intentionally sparse but useful:

- `error_code`: one of the codes above.
- `repro_note`: human summary with context and a hint.
  - Include the public API boundary, e.g. `Model::build(TensorList)` or `Graph::add(Model)`.
  - Include model path/package path when safe.
  - Include the original exception text.
- `pipeline_string`: empty when no GStreamer pipeline exists yet.
- `repro_gst_launch`: empty when no pipeline exists yet; do **not** fabricate a fake command.
- `repro_env`: optional; use only when we know a useful env var/debug mode.
- `nodes`, `bus`, `boundaries`, `caps_dump`, `dot_paths`: normally empty before build.

Avoid adding model-specific fields to `GraphReport` in Phase -1. It is a public aggregate and
changing it has ABI/API implications. Put model context in `repro_note` for now.

### Helper design that matches real `GraphReport`

Normalize at public boundaries and targeted validation points. Do **not** add fields/constructors to
`GraphReport`.

Recommended helper shape:

```cpp
struct ModelErrorContext {
  std::string_view where;
  std::string_view detail;
  std::string_view model_path;
  std::string_view hint;
};

GraphReport make_model_report(std::string_view code, const ModelErrorContext& ctx) {
  GraphReport rep{};
  rep.error_code = std::string(code);
  rep.repro_note = format_model_repro_note(ctx);  // folds where/detail/model_path/hint here
  return rep;
}

[[noreturn]] void throw_model_error(std::string_view code, const ModelErrorContext& ctx) {
  GraphReport rep = make_model_report(code, ctx);
  throw NeatError("[" + rep.error_code + "] " + rep.repro_note, std::move(rep));
}
```

Optional wrapper only for public boundaries that still need fallback normalization:

```cpp
template <class F>
decltype(auto) wrap_model_boundary(std::string_view default_code,
                                   const ModelErrorContext& ctx,
                                   F&& fn);
```

Catch policy and order:

1. `catch (const NeatError&)` -> rethrow unchanged.
2. `catch (const internal::ModelArchiveError&)` -> map via archive enum.
3. `catch (const nlohmann::json::exception&)` -> `io.model_metadata` or `io.parse`, depending on
   boundary.
4. `#if defined(SIMA_WITH_OPENCV) catch (const cv::Exception&)` -> `misconfig.model_input` for
   Model cv::Mat boundaries.
5. `catch (const std::invalid_argument&)` -> context-specific, usually `misconfig.model_options`
   or `misconfig.model_input`.
6. `catch (const std::runtime_error&)` -> context-specific fallback:
   - build/run input validation -> `misconfig.model_input`
   - planning/route fallback -> `misconfig.model_route`
   - materialization fallback -> `build.model_materialize`
   - graph composition fallback -> `misconfig.graph_compose`
7. Do **not** catch `std::bad_alloc` or unknown/fatal exceptions.

### Granularity strategy

A method-level wrapper alone cannot accurately map every inner throw. Phase -1 will therefore use a
hybrid strategy:

1. **Targeted direct conversions for known user errors**
   - Empty input lists -> `misconfig.model_input`.
   - Ingress-count mismatch -> `misconfig.model_input`.
   - Bad Model/Route options caught at option-validation points -> `misconfig.model_options`.
   - Metadata JSON parse/type errors -> `io.model_metadata`.
   - Archive typed errors -> `io.model_archive` / `io.model_extract`.
   - Model graph-composition overload errors -> `misconfig.graph_compose`.
   - Python Model binding list/tuple misuse -> `misconfig.python_input`.
2. **Boundary fallback conversions for remaining public Model failures**
   - Constructor/planner fallback -> `misconfig.model_route` unless the caught type/context says
     otherwise.
   - Graph/fragment materialization fallback -> `build.model_materialize`.
   - Build materialization fallback before downstream `Graph::build` -> `build.model_materialize`.
3. **Preserve internal helper semantics**
   - Internal helper/unit tests can keep expecting `std::invalid_argument` when they call helpers
     directly.
   - Public Model API should normalize to `NeatError`.
4. **Preserve existing `NeatError`**
   - Re-throw existing `NeatError` unchanged to avoid double-wrapping downstream Graph/session errors.

Purpose: get stable public error behavior without rewriting all 57 internal raw throws in one phase.

### Public boundaries to change

1. `Model::Model(path, options)`
2. `Model::graph(...)`, `Model::fragment(...)`, `Model::backend_fragment(...)`
3. `Model::build(...)` overloads, before and around node materialization; preserve downstream
   `Graph::build(...)` `NeatError` unchanged.
4. `Model::run(...)` overloads; empty input checks should become `misconfig.model_input`.
5. `Model::metadata()` and any introspection API that can parse/read generated files.
6. `Graph::add(const Model&)` and model-specific `Graph::connect(...)` overloads.
7. Python `Model` / `ModelRunner` binding lambdas and conversion paths.

### Python parity decision and plumbing

Decision: **full parity for Model-facing Python APIs**.

- Keep list/tuple input required.
- Do not add a new Python exception type.
- Add `error_codes::kPythonInput = "misconfig.python_input"` and export
  `pyneat.ERROR_PYTHON_INPUT`.
- Add a small binding helper in `python/src/module.cpp`, e.g.:

```cpp
[[noreturn]] void throw_python_model_input_error(const char* where,
                                                 const std::string& detail,
                                                 const std::string& hint) {
  GraphReport rep{};
  rep.error_code = error_codes::kPythonInput;
  rep.repro_note = std::string(where) + ": " + detail;
  if (!hint.empty()) rep.repro_note += "\nHint: " + hint;
  throw NeatError("[" + rep.error_code + "] " + rep.repro_note, std::move(rep));
}
```

- Use this only for `Model` / `ModelRunner` Python input conversion to avoid changing unrelated
  `Run` / `Graph` Python behavior in the same phase unless intentionally tested.
- The existing translator will expose it as `pyneat.NeatError` with `.error_code`, `.repro_note`,
  `.pipeline_string`, and synthesized `.report_json`.

### Tests to add before declaring Phase -1 done

C++ tests:

1. Invalid/missing model archive path throws `NeatError` with `io.open` or `io.model_archive`.
2. Corrupt/schema-invalid model archive throws `NeatError` with `io.model_archive`.
3. Archive extraction/storage failure maps to `io.model_extract` when practical.
4. Bad preprocess option, e.g. normalize stddev <= 0, throws `misconfig.model_options` at the public
   Model boundary.
5. Impossible/missing route stage throws `misconfig.model_route` at the public Model boundary.
6. Materialization fallback throws `build.model_materialize`.
7. `Model::build(TensorList{})`, `Model::build(Sample{})`, `Model::run(TensorList{})`, and
   `Model::run(Sample{})` throw `misconfig.model_input`.
8. Under `SIMA_WITH_OPENCV`, bad cv::Mat input throws `NeatError` with `misconfig.model_input`.
9. `metadata()` malformed JSON parse/type failures throw `io.model_metadata`.
10. `Graph::add(model)` / `Graph::connect(... model ...)` endpoint or topology failures throw
    `misconfig.graph_compose`.
11. Existing `Graph::build(...)` and runtime `Run::pull()` failures still preserve their current
    `NeatError` reports and are not double-wrapped.
12. ABI/compatibility test: a converted public Model failure is still catchable by
    `catch (const std::runtime_error&)` because `NeatError` derives from `std::runtime_error`.
13. Caller audit: any tests/callers expecting exact `std::invalid_argument` from public Model APIs
    are updated; internal helper tests can remain unchanged.

Python tests:

1. C++-origin Model failures are catchable as `pyneat.NeatError` and expose `.error_code` and
   synthesized `.report_json`.
2. `model.run(tensor)` / `model.build(tensor)` still fail because list/tuple is required, but now
   fail as `pyneat.NeatError` with `misconfig.python_input` and a hint to pass `[tensor]`.
3. `model.run(sample)` / `model.build(sample)` similarly fail as `pyneat.NeatError` with
   `misconfig.python_input` and a hint to pass `[sample]`.
4. Non-list/tuple non-DLPack input to Model/ModelRunner APIs fails as `pyneat.NeatError` with
   `misconfig.python_input`.
5. Existing non-Model `Graph` / `Run` Python conversion behavior is unchanged unless we explicitly
   choose to broaden parity later.

### Recommended implementation order

1. Add new error-code constants in `include/pipeline/ErrorCodes.h`.
2. Export new constants in Python, including `ERROR_PYTHON_INPUT`.
3. Add the aggregate-compatible Model error helper; do not modify `GraphReport` layout.
4. Preserve/remap `ModelArchiveErrorClass` before `ModelPack` erases it.
5. Wrap `Model::Model(path, options)` with correct catch ordering.
6. Convert targeted `Model::build/run` input validation throws to `throw_model_error(...)`.
7. Add `metadata()` JSON-specific catch for `io.model_metadata`.
8. Add `cv::Exception` catch around Model cv::Mat paths under `SIMA_WITH_OPENCV`.
9. Wrap model graph/fragment/backend_fragment public APIs with materialization fallback.
10. Wrap `Graph::add(model)` and model-specific `Graph::connect` composition paths.
11. Add Python Model/ModelRunner binding helper and use it for list/tuple/type conversion failures.
12. Add/update C++ tests.
13. Add/update Python tests.
14. Update docs to say Model setup/planning/build/run/composition failures throw `NeatError` only
    after tests are green.

### Risks / compatibility

- `NeatError` derives from `std::runtime_error`, so most C++ callers catching `std::runtime_error`
  keep working.
- Callers checking exact exception types may observe a behavior change at public Model boundaries.
- Python Model callers may see failures change from `RuntimeError` to `pyneat.NeatError`.
- Method-level fallback codes are intentionally coarse; exact per-throw granularity requires deeper
  refactoring and is outside the low-blast-radius part of Phase -1.
- Do not over-expand `GraphReport` yet; context in `repro_note` is enough for Phase -1.

### Non-goals for Phase -1

- Do not add `Full()` / `ModelRoute()`.
- Do not allow bare Python single tensors/samples; list/tuple remains required.
- Do not redesign `GraphReport` or add fields.
- Do not rewrite every internal raw throw.
- Do not change Model route architecture or pre/infer/post behavior.
- Do not fabricate GStreamer reproducers for pre-build failures.

### Phase -1 recommendation

Proceed with a focused error-normalization implementation before broad docs cleanup:

- implement code constants/helper/wrappers with the corrected catch order,
- preserve archive categories,
- normalize Python Model binding input errors to `pyneat.NeatError`,
- add C++/Python tests for representative failures,
- then update docs to match the now-true behavior.


## 0. Phase 0: Architecture alignment

Before fixing code, docs, or bindings, we need to lock the real SiMa Neat architecture.

The correct starting point is:

> `Model` is a simplification of `Graph` for the common customer task: "run my ML model."
>
> Customers coming from the GPU world are used to calling a model as one thing. In SiMa,
> the executable path may include pre-route transforms/adapters, one or more inference artifacts,
> post-route transforms/adapters, and optional decode. `Model` hides that graph-shaped route behind
> a model-shaped API.

So the conceptual ladder is:

```text
Model
  Simple ML-model UX.
  The user wants to call one model, even though SiMa lowers it to a route.

Graph
  Application composition layer.
  Used when the user needs cameras, branching, multiple models, custom nodes,
  named endpoints, RTSP, telemetry, or other app-level topology.

Node
  Lower-level unit used by Graph and by pre-built graph fragments.
```

The customer-facing progression should therefore be:

1. `Model::run(...)` — simplest "call the model" path.
2. `Model::build(...)` / `Model::Runner` — reusable production form of the same model route.
3. `Graph::add(model)` — put the full selected model route inside a larger app graph.
4. `model.graph(...)` — advanced: inspect/reuse/customize the model route as a graph fragment.
5. `model.preprocess()`, `model.inference()`, `model.postprocess()` — expert/debug route fragments.

### Phase 0 decisions already accepted

| Topic | Decision |
| --- | --- |
| What is `Model`? | A customer-facing ML model abstraction implemented by lowering to a SiMa graph route. |
| Why does `Model` exist? | To make SiMa's required pre/infer/post route feel like one callable ML model. |
| What is `Graph`? | The advanced application composition API. |
| What is `Graph::add(model)`? | Add the full selected model route into a larger graph. |
| What is `model.graph()`? | Return the selected model route as a reusable graph fragment. |
| What are stage fragments? | Expert-level fragments for custom wiring/debug, not the first concept users should learn. |
| What is `Infer()`? | Reserved for the inference portion of a model route that may eventually span multiple `.elf` and `.so` artifacts. It is not the full pre+infer+post route. |
| Add `Full()` / `ModelRoute()` now? | No. Use existing full-route APIs: `Model::run`, `Model::build`, `Graph::add(model)`, and `model.graph`. |
| Python bare single input? | No for now. Keep list/tuple required so single-input and multi-input models share one contract. |
| Error contract? | Add Phase -1 to investigate making setup/planning/model errors true `NeatError`s before documenting that promise. |

### Important correction about `Infer()`

`Infer()` should **not** be redefined as the full model route.

The intended future direction is that a model's inference portion may be split across multiple
compiled/runtime artifacts, for example several `.elf` and `.so` files. `Infer()` is the named
concept that should eventually run that multi-artifact inference section.

Current implementation mostly maps `Infer()`/`infer()` to today's inference fragment, but that is
a placeholder for a wider inference-stage concept, not evidence that `Infer()` should become
preprocess + inference + postprocess.

Implications:

- `Infer()` / `model.inference()` should remain inference-stage concepts.
- Full user-facing model execution should be expressed as `Model::run(...)`,
  `Model::build(...)`, `Graph::add(model)`, or `model.graph(...)`.
- Do not add a new full-route helper now. Existing full-route APIs are enough. If a future
  helper is ever proposed, it must be a separate API decision and must not reuse `Infer`.
- Docs should explain that "inference" in SiMa may be more than one artifact, while "model run"
  is the full route users usually want.

### Phase 0 deliverable

Update this plan and then the public docs so they teach:

```text
Simple path:
  Model model("foo.tar.gz");
  auto out = model.run(input);

Production model-only path:
  auto runner = model.build(input);
  auto out = runner.run(input);

Application graph path:
  Graph app;
  app.add(nodes::Input("image"));
  app.add(model);              // full selected model route
  app.add(nodes::Output("out"));

Advanced route fragment path:
  Graph route = model.graph();
```

Do **not** teach users that they must manually wire preprocess/inference/postprocess for the normal
case. That is exactly what `Model` is meant to hide.

---

## 1. Goal

Make `Model` the clear, stable high-level API for "run this ML model" usage.

The ideal contract should be simple:

```text
Model
  customer-facing ML model abstraction
  loads a compiled model package
  plans the full SiMa route needed to execute that model
  hides pre-route transforms, inference artifacts, post-route transforms, and optional decode
  behind a model-shaped API
  exposes metadata/specs for validation and introspection
  provides one-shot run() and reusable build()/Runner APIs
  can be inserted into Graph as the full selected model route

Graph
  advanced application composition API
  composes inputs, outputs, Models, Nodes, and reusable fragments

Node
  lower-level graph building block or boundary declaration

Runner
  built/live execution object for repeated model or graph execution
```

The current implementation is close in capability, but the conceptual and API boundaries are inconsistent.

---

## 2. Current architecture

`Model` currently acts as a facade over several lower-level systems:

| Layer | Main files / symbols | What it does |
| --- | --- | --- |
| Archive / package loading | `ModelPack`, `ModelArchiveLoader` | Extracts model archive, validates/loads config, exposes stage fragments. |
| Preprocess planning | `InputPlanner`, `build_preprocess_plan(...)` | Resolves user options into concrete preprocess requirements. |
| Route planning | `RoutePlanner`, `build_route_plan(...)`, `plan_route_selection(...)` | Chooses pre/infer/post route shape. |
| Graph materialization | `build_pipeline_nodes(...)`, `ModelAccess` | Turns model package stages into `Node`/`Graph` fragments. |
| Runtime | `Model::Runner`, `Run` | Builds and executes the pipeline. |
| Python API | `python/src/module.cpp` | Nanobind wrapper for a subset of the C++ API. |

Useful entry points:

- Public C++ API: `include/model/Model.h`
- Main implementation: `src/model/Model.cpp`
- Route planner: `src/model/RoutePlanner.cpp`, `src/model/internal/RoutePlanner.h`
- Package loader: `src/model/ModelPack.cpp`, `src/model/ModelArchiveLoader.cpp`
- Python binding: `python/src/module.cpp`

---

## 3. High-level diagnosis

The main problem is not a single bug. It is drift between four things:

1. The C++ public API.
2. The Python public API.
3. The docs/tutorials.
4. The actual implementation behavior.

Right now, these do not describe exactly the same product.

The biggest categories are:

- API contract mismatch.
- Docs/tutorials stale against implementation.
- Python binding incomplete relative to C++.
- Duplicated route decision logic.
- A few likely correctness bugs in graph materialization / BoxDecode metadata.
- Weak edge-case test coverage.

---

## 4. Detailed findings

### 4.1 Error contract mismatch: docs/header say `NeatError`, implementation throws raw exceptions

**Evidence**

The public API documentation in `include/model/Model.h` implies `Model` failures should surface as `NeatError` in many paths.

But the implementation commonly throws:

- `std::runtime_error`
- `std::invalid_argument`
- JSON parser exceptions
- archive errors wrapped as `std::runtime_error`

Examples:

- `src/model/Model.cpp:3989` throws `std::runtime_error` for unresolved preprocess resize.
- `src/model/Model.cpp:6785-6808` one-shot `run(...)` throws `std::runtime_error` on empty input.
- `src/model/ModelPack.cpp:1108-1110` catches `ModelArchiveError` and rethrows `std::runtime_error("ModelPack: ...")`.
- `Model::metadata()` parses JSON directly and can surface JSON exceptions.

**Why this matters**

A user reading the docs may catch/handle `NeatError`, but construction, planning, metadata, and package-loading paths can bypass that contract.

**Decision needed**

Pick one:

1. Normalize `Model` to throw/report `NeatError` consistently.
2. Update docs to say `Model` can throw standard exceptions during construction/planning and reserves `NeatError` for graph/runtime diagnostics.

**Recommendation**

Short term: fix docs honestly.
Long term: design one consistent exception/diagnostic model.

---

### 4.2 Route planning has two sources of truth

**Evidence**

`build_preprocess_plan(...)` computes both:

```cpp
internal::SessionRoutePlan session_route_plan =
    internal::build_route_plan(options, model_semantics, &capability, &capability_pack);
const internal::RouteSelection route =
    internal::plan_route_selection(options, plan, capability);
```

See `src/model/Model.cpp:3610-3615`.

Then:

- `plan.session_route_plan` drives materialization.
- `route` feeds diagnostics and selected compatibility fields.

Examples:

- `src/model/Model.cpp:3740-3748`
- `src/model/Model.cpp:3828-3830`

**Why this matters**

Two route decision engines can diverge. If they disagree, the actual graph can do one thing while diagnostics say another.

**Recommendation**

Make `SessionRoutePlan` the single source of truth. If `RouteSelection` is still needed, derive it from the finalized `SessionRoutePlan`, not independently from options/capability.

---

### 4.2A Verified: `Graph::add(model)` and `model.graph()` lowering behavior

**Evidence**

`Model::RouteOptions` defaults to no public boundary nodes:

```cpp
bool include_input = false;
bool include_output = false;
```

See `include/model/Model.h:282-284`.

`Graph::add(const Model&)` explicitly creates those default route options and appends
`model.graph(opt)`:

```cpp
Model::RouteOptions opt;
opt.include_input = false;
opt.include_output = false;
return add(model.graph(opt));
```

See `src/pipeline/graph/GraphModel.cpp:7-13`.

`Model::graph()` calls `ModelAccess::build_graph_fragment(...)`, which builds route nodes and
attaches model boundary hints/provenance:

- route nodes: `src/model/Model.cpp:7170-7174`
- ingress endpoint names: `src/model/Model.cpp:7175-7194`
- egress endpoint names: `src/model/Model.cpp:7195-7199`
- input-route processor for bundled multi-input fan-in: `src/model/Model.cpp:7207-7210`
- route provenance: `src/model/Model.cpp:7215-7233`

`Graph::add(Graph)` preserves fragment metadata when appending a linear fragment:

- fragment plans rebased: `src/pipeline/graph/Graph.cpp:1029-1031`
- input route processor carried across: `src/pipeline/graph/Graph.cpp:1042-1048`

`Graph::connect(... Model ...)` imports `model.graph()` and uses model-derived endpoint names for
endpoint matching:

- import/reuse model fragment: `src/pipeline/graph/Graph.cpp:1311-1330`
- model connect overloads: `src/pipeline/graph/Graph.cpp:1574-1616`

**Verified behavior**

- `Graph::add(model)` is the right common application-composition API.
- It does **not** insert public `Input`/`Output` nodes by default.
- It splices the selected full model route into the surrounding graph.
- `model.graph()` is the same selected route as a reusable `Graph` fragment with boundary hints.
- `model.graph({include_input=true, include_output=true})` is the standalone/runnable route form.
- Multi-input model routes rely on boundary hints and the input-route processor during connected
  graph lowering; this is already wired through the compiler path.

**Docs implication**

Use `Graph::add(model)` for normal app composition docs. Use `model.graph()` when teaching advanced
fragment inspection/reuse, standalone route construction, or explicit `RouteOptions`.

---

### 4.3 Verified bug: preprocess upstream can point to itself

**Evidence**

In `build_pipeline_nodes(...)`:

```cpp
std::string upstream =
    popt.upstream_name.empty() ? (pre_name.empty() ? "decoder" : pre_name) : popt.upstream_name;
```

See `src/model/Model.cpp:5635-5636`.

Then preprocess is built with:

```cpp
const std::string pre_upstream =
    (popt.include_input && !src_opt.buffer_name.empty()) ? src_opt.buffer_name : upstream;
auto pre_nodes =
    build_preprocess_nodes_impl(model, pack, plan, input, pre_name, pre_upstream, sync);
```

See `src/model/Model.cpp:5642-5646`.

When preprocess exists and there is no explicit upstream/buffer, `upstream` becomes `pre_name`. That means the preprocess fragment can be asked to attach to itself.

Standalone preprocess uses the expected default upstream of `decoder` elsewhere.

**Why this matters**

This can create incorrect graph linkage for full-route model graphs that include preprocess.

**Verification**

This is now verified by code inspection, not just suspected.

For a full route with preprocess and default route options:

1. `pre_name` is set to the resolved preprocess stage name.
2. `upstream` becomes `pre_name` when `popt.upstream_name` is empty.
3. If no explicit input buffer name is present, `pre_upstream` also becomes `pre_name`.
4. `PreprocOptions.upstream_name` is serialized into the generated preproc JSON
   `input_buffers[0].name`.

Relevant code:

- upstream selection: `src/model/Model.cpp:5635-5636`
- preprocess upstream selection: `src/model/Model.cpp:5642-5646`
- standalone preprocess uses `decoder`: `src/model/Model.cpp:7055-7062`
- preproc JSON writes `input_buffers[0].name = opt.upstream_name`:
  `src/nodes/sima/Preproc.cpp:289-292`

So a route whose preprocess element is named, for example, `preproc_0` can generate a preproc
config saying its input buffer is also `preproc_0`. The preprocess stage should instead consume the
external/source upstream (`decoder`, route `upstream_name`, or the actual `Input` buffer name), while
inference should consume the preprocess output.

**Recommendation**

Define two separate names:

```text
source_upstream = decoder/input/external upstream
infer_upstream  = preprocess output when preprocess is included, otherwise source_upstream
```

Then use:

- preprocess attaches to `source_upstream`
- inference attaches to `infer_upstream`

Add a regression test that asserts no pre-stage receives its own name as upstream.

---

### 4.4 BoxDecode metadata path is internally contradictory

**Evidence**

`make_preprocess_meta_template(...)` has this early return:

```cpp
if (plan.session_route_plan.boxdecode_selected &&
    rp.graph_family != PreprocessGraphFamily::Preproc) {
  return std::nullopt;
}
```

See `src/model/Model.cpp:4106-4117`.

But immediately after that, there is a comment and code path saying adapter-only ingress may still need preprocess metadata for BoxDecode:

```cpp
// Adapter-only ingress (quant/tess/quanttess) may still feed boxdecode and
// therefore must carry preprocess runtime metadata even without a preproc graph.
```

See `src/model/Model.cpp:4123-4143`.

That adapter-only path is unreachable when BoxDecode is selected and the graph family is not `Preproc`.

**Verification**

This is verified by code inspection:

- `InputOptions.preprocess_meta` is the route-level mechanism that causes ingress buffers to be
  annotated with `PreprocessRuntimeMeta`.
- `apply_simaai_preprocess_meta_template(...)` writes the original input width/height and affine
  fields from that template at push time (`src/pipeline/gst/InputStreamUtil.cpp:2626-2726`).
- `BoxDecodeSample(...)` explicitly reads preprocess metadata and rejects missing
  `preproc_original_width/preproc_original_height` (`src/pipeline/runtime/StageRun.cpp:2582-2651`).
- `SimaBoxDecode` carries required preprocess metadata fields unless explicit original/model
  dimensions relax them (`src/nodes/sima/SimaBoxDecode.cpp:365-411`, `:487-511`, `:935-948`).
- Therefore, returning `std::nullopt` from `make_preprocess_meta_template(...)` for non-Preproc
  BoxDecode routes removes the metadata source that later BoxDecode paths require.

The code even says adapter-only ingress may need this metadata, but the early return prevents that
branch from running.

**Why this matters**

BoxDecode often needs original image dimensions / preprocess metadata to map boxes correctly. Non-Preproc adapter-only routes may fail later or decode boxes incorrectly.

**Recommendation**

Split the conditions:

- If BoxDecode selected and no Preproc graph, still generate minimal metadata when route/input contracts provide enough information.
- Only return `nullopt` when metadata is truly unnecessary or impossible.

Add targeted tests for:

- BoxDecode with Preproc route.
- BoxDecode with adapter-only route.
- BoxDecode with no detections.
- BoxDecode with original width/height override.

---

### 4.5 Python binding lags behind C++ `Model`

**Evidence**

C++ exposes a larger API in `include/model/Model.h`, but Python only binds part of it in `python/src/module.cpp:2902-3098`.

Missing or incomplete Python exposure includes:

| C++ API / field | Status in Python |
| --- | --- |
| `input_specs()` | Missing; only `input_spec()` exists. |
| `output_specs()` | Missing; only `output_spec()` exists. |
| `compiled_batch_size()` | Missing. |
| `resolved_preprocess_plan()` | Missing. |
| `preprocess_requirements()` | Missing. |
| `info()` / `ModelInfo` | Missing. |
| `input_appsrc_options_list(...)` | Missing; only singular accessor exists. |
| `Model::Options::prepared_runner` | Missing. |
| `Model::Options::async_queue_depth` | Missing. |
| Several `RouteOptions` fields | Missing. |
| `Runner` diagnostics/stats/report/metrics APIs | Missing. |
| no-input `build(RouteOptions, RunOptions)` variants | Incomplete. |

Current Python bindings:

- `ModelOptions`: `python/src/module.cpp:2902-2918`
- `ModelRouteOptions`: `python/src/module.cpp:2920-2927`
- `ModelRunner`: `python/src/module.cpp:2935-3006`
- `Model`: `python/src/module.cpp:3008-3098`

**Why this matters**

Python users cannot inspect or configure models at the same level as C++ users. Multi-input models are especially problematic because C++ single-input helpers intentionally reject multi-ingress use, but Python lacks the plural alternatives.

**Recommendation**

Python `Model` should match C++ `Model` capabilities unless there is a deliberate, documented
Python-specific reason not to expose something.

Binding parity targets:

1. Introspection:
   - `input_specs()`
   - `output_specs()`
   - `compiled_batch_size()`
   - `info()` / `ModelInfo` or a faithful dict equivalent
   - `resolved_preprocess_plan()`
   - `preprocess_requirements()`
   - `input_appsrc_options_list(...)`
2. Options parity:
   - `ModelOptions.prepared_runner`
   - `ModelOptions.async_queue_depth`
   - all `ModelRouteOptions` fields currently missing from Python:
     `verbose`, `processcvu_requested_run_target`, `processcvu`, `processmla`,
     `prepared_runner`, `async_queue_depth`.
3. Build parity:
   - expose `build(RunOptions)` and `build(RouteOptions, RunOptions)` cleanly in Python, not only
     `build()` / `build_with_route_options(...)` and seeded `build(input, ...)`.
4. Runner parity:
   - `start_measurement(...)`
   - `stats()`
   - `measurement_summary()`
   - `metrics(...)`
   - `metrics_report(...)`
   - `diag_snapshot()`
   - `report(...)`
   - `close_input()`
5. Tests:
   - add a Python parity test that asserts every intended public C++ Model capability is either
     exposed in Python or listed in an explicit denylist with rationale.

---

### 4.6 Docs/tutorials are stale or misleading

**Evidence**

Several examples do not match current APIs.

Known issues:

- Some Python tutorials use obsolete flattened options instead of current nested preprocess options.
- Some set `decode_type` using string values like `"yolov8"`; binding expects enum-style values.
- Some tutorials treat tensor-list `model.run([tensor])` as returning a `Sample`, but tensor input returns `TensorList`.
- Docs show bare `model.run(arr)` / `model.run(torch_tensor)`, but the binding rejects a single tensor/sample and expects list/tuple input.
- Header examples use `sima::Model`, while code namespace is `simaai::neat`.
- Some tutorial titles imply they use `Model`, but the code only demonstrates Tensor/Graph interop.

Examples:

- `tutorials/001_run_your_first_model/run_your_first_model.py:46`
- `tutorials/004_configure_model_options/configure_model_options.py:54`
- `tutorials/005_preprocess_images/preprocess_images.py:51`
- `tutorials/006_read_detection_boxes/read_detection_boxes.py:52`
- `tutorials/016_build_production_pipeline/build_production_pipeline.py:52`
- `python/src/module.cpp:3077-3098` shows current `Model.run` Python behavior.

**Why this matters**

Users will copy broken examples. CI can still pass because Python tutorials appear to be syntax-checked more than executed end-to-end.

**Recommendation**

Treat docs/tutorial cleanup as a first-class API task, not polish.

Fix order:

1. Correct `Model.run` return-type examples.
2. Correct Python input wrapping examples.
3. Correct `ModelOptions` / `PreprocessOptions` examples.
4. Rename or rewrite tutorials that do not actually use `Model`.
5. Add a small executable smoke test for the canonical Python tutorial path.

---

### 4.7 `ModelGroups::Infer` future contract needs clearer docs

**Evidence**

Header says the grouped-node helper wraps:

```text
preprocess, MLA, postprocess, and the combined Infer fragment that chains them together
```

See `include/nodes/groups/ModelGroups.h:4-8`.

But implementation returns only inference/MLA:

```cpp
simaai::neat::Graph infer(const std::string& tar_gz) {
  simaai::neat::Model model(tar_gz, simaai::neat::Model::Options{});
  return model.inference();
}

simaai::neat::Graph Infer(const simaai::neat::Model& model) {
  return model.inference();
}
```

See `src/nodes/groups/ModelGroups.cpp:73-80` and `src/nodes/groups/ModelGroups.cpp:95-97`.

**Why this matters**

The original investigation treated this as a possible bug because `Infer` can sound like the full
model route. That was the wrong architectural assumption.

In the intended architecture, `Infer()` is reserved for the inference section of a model route.
That inference section may eventually span multiple `.elf` and `.so` files. So `Infer()` is not
supposed to mean the full pre+infer+post route.

The real problem is documentation clarity:

- full model execution should be described as `Model::run(...)`, `Model::build(...)`,
  `Graph::add(model)`, or `model.graph(...)`;
- `Infer()` / `model.inference()` should be described as the inference-stage fragment;
- current `Infer()` implementation may be incomplete relative to the future multi-artifact
  inference-stage goal, but it should not be changed to return the full model route.

**Decision needed**

Clarify the naming contract:

1. `Infer()` means the inference-stage route, currently mostly MLA-only, future multi-`.elf` /
   multi-`.so`.
2. Full model route means `model.graph(...)`, `Graph::add(model)`, `Model::run(...)`, or
   `Model::build(...)`.
3. Do not add a named full-route group now. Existing APIs are enough. Do not overload
   `Infer()`.

**Recommendation**

Keep explicit names:

- `Infer(model)` = inference-stage route; future home for split `.elf`/`.so` inference.
- `MLA(model)` = single MLA-stage helper where that distinction is useful.
- `Preprocess(model)` = pre-stage only.
- `Postprocess(model)` = post-stage only.
- `model.graph()` / `Graph::add(model)` = complete selected model route.
- No new full-route group helper now; use existing full-route APIs.

Do not repurpose `Infer()` as full pre+infer+post.

---

### 4.8 Archive extraction can collide on basename

**Evidence**

Archive validation rejects duplicate normalized archive paths:

- `src/model/ModelArchiveLoader.cpp:914-924`

But extraction flattens entries to filename-only destinations:

```cpp
const fs::path name = fs::path(entry.normalized_path).filename();
return package_root / "etc" / name;
```

See `src/model/ModelArchiveLoader.cpp:995-1009`.

The write happens here:

- `src/model/ModelArchiveLoader.cpp:1088-1089`

So an archive with:

```text
a/config.json
b/config.json
```

can pass normalized-path duplicate checks but collide in the flattened destination.

Also, `ModelPack` accepts a directly extracted directory and bypasses archive validation:

- `src/model/ModelPack.cpp:1037-1043`

**Why this matters**

Malformed archives can overwrite files during extraction or behave differently depending on archive-vs-directory loading.

**Recommendation**

Add validation for destination-path collisions after applying `extract_destination_for(...)`.

For direct extracted directories, either:

- document that this is a trusted developer path, or
- add a lightweight validation pass for expected layout/config files.

---

### 4.9 Cleanup option is effectively global, not truly per-model

**Evidence**

Public option:

- `Model::Options::cleanup_extracted_model_data`

Implementation uses process-global state:

- `src/model/ModelPack.cpp:711-719`
- `src/model/ModelPack.cpp:807-823`
- `src/model/ModelPack.cpp:784-798`

Extraction cache key does not include cleanup behavior:

- `src/model/ModelPack.cpp:1025-1047`
- cache hit early return: `src/model/ModelPack.cpp:1055-1061`

**Why this matters**

If one load extracts with cleanup enabled, and a later load requests cleanup disabled, the later load can hit the cache and return early without marking the process root as keep/disabled. User intent can depend on load order.

**Recommendation**

Short term: document cleanup as process-level behavior.
Better: make cache handling honor the most conservative request. If any request says `cleanup_extracted_model_data=false`, mark the cached root as keep and disable cleanup for that root/process.

---

### 4.10 Thread-safety is unclear

**Evidence**

`Model::Impl` has synchronization fields:

```cpp
mutable std::mutex sync_mu;
mutable bool sync_ready = false;
mutable InputKey sync_key{};
mutable Runner sync_runner{};
```

See `src/model/Model.cpp:3855-3858`.

But lazy sync pack initialization mutates without visible locking:

```cpp
const internal::ModelPack& pack_for_sync() const {
  if (!sync_pack.has_value()) {
    sync_pack = pack.clone_with_buffers(1, 1);
  }
  return *sync_pack;
}
```

See `src/model/Model.cpp:4085-4090`.

There are also route processors that appear to hold mutable branch runners and build/run/reset them in runtime paths.

**Why this matters**

Sharing `Model` or Model-derived graph fragments across threads may race unless users serialize externally.

**Recommendation**

Pick/document a contract:

1. `Model` construction and introspection are thread-safe; building/running is not.
2. Or make `Model` fully thread-safe for concurrent `build/run` by locking lazy state.

At minimum, protect `sync_pack` lazy initialization.

---

### 4.11 Model-level output introspection loses semantic information

**Evidence**

`SimaBoxDecode` can advertise BBOX output spec:

- `src/nodes/sima/SimaBoxDecode.cpp`

But `Model::output_specs()` for BoxDecode appears to return a generic unknown/rank-ish spec rather than semantic BBOX output.

**Why this matters**

Users cannot reliably inspect from `Model` that output is decoded boxes.

**Recommendation**

Model-level output specs should preserve known semantic output type where possible:

- decoded boxes
- tensor list
- multi-output logical names
- shape/dtype where available

---

### 4.12 `Model::run(...)` is convenience, not cheaper execution

**Evidence**

Header says:

```text
Equivalent to build(inputs).run(inputs, timeout_ms) but cheaper because the Runner is scoped to the call.
```

See `include/model/Model.h` around the `run(...)` docs.

Implementation literally does:

```cpp
Runner runner = build(inputs);
return runner.run(inputs, timeout_ms);
```

See `src/model/Model.cpp:6785-6808`.

**Why this matters**

One-shot `run(...)` is fine for examples, but expensive for repeated inference. The docs should not imply it is cheaper than using a long-lived runner.

**Recommendation**

Document:

- `Model::run(...)` = convenience one-shot, good for smoke tests and simple scripts.
- `Model::build(...)` + reused `Runner` = production/repeated inference path.

---

### 4.13 Model ID/provenance is extraction-path based

**Evidence**

`model_id` is based on `pack.etc_dir()`:

```cpp
model_id = pack.etc_dir();
if (!options.name_suffix.empty()) {
  model_id += "::" + options.name_suffix;
}
```

See `src/model/Model.cpp:4079-4082`.

**Why this matters**

The same archive can get a different model ID depending on extraction root/process. This weakens diagnostics/provenance/cache stability.

**Recommendation**

Use a stable package identity where possible:

- archive path + mtime/size
- archive content hash
- manifest model name/version
- suffix/options hash if needed

---

## 5. Missing test coverage

Important gaps:

- Python tutorial files are mostly syntax-compiled, not executed against real model fixtures.
- No binding parity test comparing C++ exposed model concepts to Python-exposed concepts.
- No test for `input_specs()` / `output_specs()` in Python because they are not bound.
- No test for route-planner divergence.
- No test for preprocess upstream self-link.
- No test for BoxDecode adapter-only metadata.
- No test for archive basename collision.
- No test for cleanup cache semantics.
- No explicit thread-safety contract test or doc test.
- Weak tests around model-level semantic output specs.

---

## 6. Proposed solution strategy

Do not try to fix everything in one huge patch. Split this into safe layers.

### Phase -1: NeatError contract investigation

Goal: determine how to make `Model` setup/planning/materialization failures consistently throw
`NeatError` with useful `GraphReport` payloads.

Tasks:

- Inventory current raw exceptions from `Model`, `ModelPack`, `InputPlanner`, `RoutePlanner`, and
  model-related Graph composition/lowering.
- Define error-code taxonomy for model setup failures.
- Design report helpers that work before a GStreamer pipeline exists.
- Decide public wrapping boundaries and Python exception behavior.
- Add representative tests before changing the docs promise.

Why before docs: the headers currently over-promise `NeatError`. We should either make that true or
change the promise deliberately after this investigation.

---

### Phase 0: Architecture alignment

Goal: ensure every later change follows the real product architecture.

Tasks:

- Treat `Model` as the simple "run my ML model" abstraction, not just another fragment factory.
- Treat `Graph` as the advanced application composition layer.
- Teach `Graph::add(model)` as the common way to embed the full selected model route.
- Teach `model.graph(...)` as the advanced reusable route-fragment API.
- Keep `Infer()` as the inference-stage concept, including the future split `.elf`/`.so`
  inference route.
- Do not redefine `Infer()` as full pre+infer+post.
- Identify docs/code that accidentally force users to think in pre/infer/post stages for the
  normal model-only path.

Why first: the previous investigation mixed up `Infer()` and full model route semantics. We should
not patch code from that wrong mental model.

---

### Phase 1: Align docs with current truth

Goal: stop teaching wrong behavior.

Tasks:

- Fix `Model.run` return-type examples.
- Fix Python list/tuple input examples.
- Fix stale `ModelOptions` examples.
- Clarify one-shot `run(...)` vs reusable `Runner`.
- Clarify current error behavior.
- Clarify `Infer()` as inference-stage/future multi-artifact inference, not full model route.
- Make `Model::run(...)`, `Model::build(...)`, `Graph::add(model)`, and `model.graph(...)`
  the canonical full-model-route APIs.

Why first: low risk, immediately reduces user confusion.

---

### Phase 2: Add regression tests around verified/suspected correctness bugs

Goal: lock behavior before changing internals.

Tests to add:

1. Full-route graph with preprocess does not wire preprocess to itself.
2. BoxDecode adapter-only/non-Preproc route carries required original-dimension metadata or fails early with a clear error.
3. Archive extraction rejects destination basename collisions.
4. Cleanup-disabled load after cleanup-enabled cache hit still preserves extracted root.

Why second: these are behavior-sensitive. Tests should lock expected behavior before refactor.

---

### Phase 3: Fix correctness bugs

Likely changes:

1. Split `source_upstream` from `infer_upstream` in `build_pipeline_nodes(...)`.
2. Repair `make_preprocess_meta_template(...)` so adapter-only BoxDecode metadata path is reachable.
3. Add extraction destination collision validation.
4. Make cleanup cache semantics conservative.
5. Protect lazy `sync_pack` initialization or document non-thread-safety.

---

### Phase 4: Python parity pass

Goal: Python `Model` should match C++ `Model` capabilities perfectly unless a difference is
explicitly documented and tested.

Required work:

1. Bind missing introspection APIs:
   - `input_specs()`
   - `output_specs()`
   - `compiled_batch_size()`
   - `info()` / `ModelInfo` or faithful Python dict
   - `resolved_preprocess_plan()`
   - `preprocess_requirements()`
   - `input_appsrc_options_list(...)`
2. Bind missing option fields:
   - `ModelOptions.prepared_runner`
   - `ModelOptions.async_queue_depth`
   - `ModelRouteOptions.verbose`
   - `ModelRouteOptions.processcvu_requested_run_target`
   - `ModelRouteOptions.processcvu`
   - `ModelRouteOptions.processmla`
   - `ModelRouteOptions.prepared_runner`
   - `ModelRouteOptions.async_queue_depth`
3. Bind missing no-input build overloads with `RunOptions` and `(RouteOptions, RunOptions)`.
4. Bind missing `ModelRunner` diagnostics/performance APIs:
   - `start_measurement`, `stats`, `measurement_summary`, `metrics`, `metrics_report`,
     `diag_snapshot`, `report`, `close_input`.
5. Add Python API parity tests with an explicit denylist for any intentionally unbound C++ API.

---

### Phase 5: Collapse route-planner duplication

Goal: one source of truth.

Approach:

- Decide that `SessionRoutePlan` is the authoritative route.
- Make diagnostic-only route selection derive from `SessionRoutePlan`.
- Remove or isolate legacy `RouteSelection` if possible.
- Update resolved plan fields from the authoritative route only.

This should happen after tests exist because it is the highest regression risk.

---

## 7. Suggested priority board

### P-1: error-contract investigation

- [ ] Inventory Model/Graph setup failures that currently throw raw exceptions.
- [ ] Design `NeatError`/`GraphReport` wrapping for setup-time failures.
- [ ] Decide Python exception/report exposure.
- [ ] Add representative `NeatError` tests before changing docs to promise it.

### P0: user-facing correctness / confusion

- [ ] Document the accepted architecture: `Model` is the simple callable ML-model abstraction over a SiMa route.
- [ ] Fix docs/tutorial examples for `Model.run` return type.
- [ ] Fix Python examples to pass list/tuple input.
- [ ] Fix stale Python option examples.
- [ ] Clarify one-shot `run` vs reusable `Runner`.
- [ ] Document `Infer()` as inference-stage/future multi-`.elf`/`.so` route, not full model route.
- [ ] Prefer `Graph::add(model)` for common app composition examples; reserve `model.graph()` for advanced route-fragment usage.

### P1: verified/suspected runtime correctness bugs

- [ ] Test and fix preprocess upstream self-link.
- [ ] Test and fix BoxDecode adapter-only metadata.
- [ ] Reject archive extraction destination collisions.
- [ ] Fix cleanup-cache semantics.

### P2: API completeness and quality

- [ ] Add Python plural specs and introspection APIs.
- [ ] Add missing Python `Options` and `RouteOptions` fields.
- [ ] Add Runner diagnostics/report bindings.
- [ ] Improve model-level semantic output specs.

### P3: architecture cleanup

- [ ] Collapse route planning to one source of truth.
- [ ] Normalize error handling.
- [ ] Define thread-safety contract.
- [ ] Stabilize model identity/provenance.

---

## 8. Open decisions for us

### Decision A: What should `Infer` mean?

Accepted direction:

`Infer()` is the inference-stage concept. It is reserved for the future where a model's inference
section can be split across multiple `.elf` and `.so` files. It is not the full model route.

Current implementation returning the existing inference fragment is acceptable as today's limited
implementation. The missing work is future expansion to multi-artifact inference, not changing
`Infer()` to pre+infer+post.

Full model route APIs are:

- `Model::run(...)`
- `Model::build(...)`
- `Graph::add(model)`
- `model.graph(...)`

Do not add `Full()` or `ModelRoute()` now. Existing full-route APIs are enough.

---

### Decision B: Python input shape for `Model.run/build`

Accepted direction: keep list/tuple required for now.

```python
model.run([tensor])      # yes
model.run(tensor)        # no, rejected intentionally for now
```

Reason: this keeps single-input and multi-input models under one explicit contract. Docs/tutorials
must show list/tuple input everywhere.

---

### Decision C: Should Python expose exact C++ types or summaries?

For complex model planning types, exact class bindings may be heavy.

Options:

1. Bind full C++ structs.
2. Expose Python dict summaries.
3. Expose JSON/debug strings first, full types later.

Recommendation: expose stable dict summaries first for `resolved_preprocess_plan()` and `preprocess_requirements()`, plus full bindings only where the type is already simple/stable.

---

### Decision D: Should `Model` throw `NeatError` everywhere?

Options:

1. Convert construction/planning/archive errors into `NeatError`.
2. Document that these are standard exceptions.
3. Hybrid: standard exceptions for setup/config, `NeatError` for runtime graph diagnostics.

Recommendation: hybrid is probably most realistic short-term, but docs must be explicit.

---

### Decision E: Should direct extracted directories be validated?

Options:

1. Trust direct directories as developer/debug path.
2. Add lightweight validation.
3. Require archive validation always.

Recommendation: keep as trusted developer path but document it, and add optional validation/debug warning.

---

## 9. First patch set I recommend

If we want the safest sequence, I recommend this first PR/patch set:

1. Complete Phase -1 error-contract investigation and decide the `NeatError` wrapping boundary.
2. Update the docs architecture language so `Model` is the simple callable ML-model abstraction
   over a SiMa route.
3. Correct `Infer()` docs: inference-stage/future multi-`.elf`/`.so`, not full pre+infer+post.
4. Fix docs/tutorials that are objectively stale, including Python list/tuple input.
5. Add a small test for Python tutorial-style `Model.run([tensor])` return type.
6. Add graph-materialization test for preprocess upstream names.
7. Add archive extraction collision test.
8. Fix only the smallest confirmed bug from those tests.

This lets us move from investigation to controlled changes without destabilizing route planning.

---

## 10. Notes from subagent split

The investigation was split roughly as:

- C++ public API and docs/tutorial consistency.
- C++ implementation and route/package internals.
- Python binding and Python tutorial/test behavior.

Common conclusion across all three: `Model` is the right high-level abstraction, but it needs API/docs/binding convergence before deeper refactors.
